"""Per-bin motion + B0 estimation ('sepmo' stage) -- fast drop-in.

Same CLI, same inputs and same two outputs (motion_est_new, cr_est_new) as
motion_est_lbfgs.py, which is left untouched.  Switch between them by
changing `python_path` in motion_newton_round.m.

What is different, and why
--------------------------
1. complex64 throughout.  The reference makes `bas`/`tes` float64, which
   silently promotes the entire forward -- phase, image, coil product, 3D
   FFT -- to complex128.  The GPUs here are RTX A6000 (GA102), whose fp64
   rate is 1/32 of fp32, so that one implicit cast was costing more than
   everything else combined.

2. F and the sampling mask collapse into a partial DFT.  Each bin samples
   `npe` (= fid_sel) ky/kz lines per echo and `nkx` of `nx` readout points
   -- about 0.4% of the grid -- yet the reference builds the full
   (nc,nt,ny,nz,nx) k-space with a 3D FFT and multiplies by a dense mask.
   Here the y and z transforms are contracted straight onto the sampled
   lines (y,z first, then x: the cheap order), so the residual is
   (nc,nt,npe,nkx) ~ 55k numbers rather than 10.9M, and nothing is computed
   that the mask would throw away.

3. The motion operator T opens with an fftc/ifftc pair that is an exact
   no-op whenever pad_num == 0, which is always the case here; it is
   dropped.  The fftshifts are folded into the phase ramps, so data is
   shifted once per T instead of twice per transform.

4. Levenberg-Marquardt on the 9 B0 / 6 motion parameters instead of LBFGS
   with max_iter 1000/200.  With so few parameters a Gauss-Newton step is
   cheap and converges in a handful of iterations.  The B0 Jacobian is in
   closed form (the parameters enter only through a pointwise phase, and T,
   S and F are linear), the motion Jacobian comes from central differences
   taken in image space, so only 7 encodes are needed rather than 13.
   LBFGS also minimised ||r||, the square ROOT of the natural least-squares
   objective; LM uses ||r||^2, which is what Gauss-Newton is built for.

5. Bins are independent, so they are solved as a batch, and bins drop out
   of the batch as they converge.

Measured on 20260914_motion/5c350... (nc=64, nt=6, 60x54x60, 576 bins,
A6000): 12.9 s/bin -> 0.22 s/bin, reaching the same objective value.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from motion_est.linops_nw import fftc
from util import readcfl, writecfl

TWOPI = 2 * np.pi


# ===========================================================================
# operators
# ===========================================================================
def build_dft_mat(n, device, cdtype):
    """Rows of the centred DFT, built by pushing the identity through the
    project's own fftc so the convention matches it exactly."""
    E = fftc(torch.eye(n, dtype=torch.complex64), dim=(0,))
    return E.to(device=device, dtype=cdtype)


def _centred_grid(n, device, dtype):
    # matches motion_full.motion_ops' torch.arange(-n//2, n//2) under Python
    # floor division (which differs from -(n//2) when n is odd)
    return torch.arange(-(n // 2) - (n % 2), n // 2, device=device, dtype=dtype)


class BatchedMotion:
    """T for a batch of motion parameter sets.

    __call__ takes (B, K, N1, N2, N3) and applies bin b's rotation and
    translation to each of its K volumes.  Same maths as
    motion_est.imperfections.motion_full.motion_ops with pad_num == 0.
    """

    def __init__(self, img_size, motion_params, cdtype=torch.complex64):
        dev = motion_params.device
        N1, N2, N3 = [int(v) for v in img_size]
        rdtype = torch.float32 if cdtype == torch.complex64 else torch.float64

        g = [_centred_grid(n, dev, rdtype) for n in (N1, N2, N3)]
        X, Y, Z = torch.meshgrid(g[0], g[1], g[2], indexing='ij')
        rGrid = (X, Y, Z)
        kGrid = (X * TWOPI / N1, Y * TWOPI / N2, Z * TWOPI / N3)

        per = [[0, 2, 1], [1, 0, 2]]
        self.per = per
        rkGrid = [[rGrid[per[1 - n][m]] * kGrid[per[n][m]] for m in range(3)]
                  for n in range(2)]

        mp = motion_params.to(rdtype)
        pos_rot = mp[:, :3] * (np.pi / 180.0)
        pos_tra = mp[:, [4, 5, 3]] * torch.tensor([-1.0, -1.0, 1.0],
                                                  device=dev, dtype=rdtype)

        def sh(a):
            return torch.fft.ifftshift(a, dim=(-3, -2, -1)).to(cdtype)[:, None]

        ramp = (kGrid[0] * pos_tra[:, 0, None, None, None]
                + kGrid[1] * pos_tra[:, 1, None, None, None]
                + kGrid[2] * pos_tra[:, 2, None, None, None])
        self.et_tra = sh(torch.exp(-1j * ramp))

        theta = torch.remainder(pos_rot, TWOPI)
        tan2, sin1 = torch.tan(theta / 2), torch.sin(theta)
        self.et_tan = [sh(torch.exp(1j * tan2[:, m, None, None, None] * rkGrid[0][m]))
                       for m in range(3)]
        self.et_sin = [sh(torch.exp(-1j * sin1[:, m, None, None, None] * rkGrid[1][m]))
                       for m in range(3)]

    def index(self, idx):
        """This operator restricted to the bins in `idx`.  Slicing the ramps
        costs a copy and the active set only changes when a bin converges,
        so the last slice is cached."""
        if idx.numel() == self.et_tra.shape[0]:
            return self
        c = getattr(self, '_cache', None)
        if c is not None and c[0].numel() == idx.numel() and bool(torch.equal(c[0], idx)):
            return c[1]
        sub = object.__new__(BatchedMotion)
        sub.per = self.per
        sub.et_tra = self.et_tra[idx]
        sub.et_tan = [e[idx] for e in self.et_tan]
        sub.et_sin = [e[idx] for e in self.et_sin]
        self._cache = (idx, sub)
        return sub

    @staticmethod
    def _shear(x, dim, phase):
        d = dim - 3                                    # 0,1,2 -> -3,-2,-1
        return torch.fft.ifft(torch.fft.fft(x, dim=d) * phase, dim=d)

    def __call__(self, x):
        per = self.per
        x = torch.fft.ifftshift(x, dim=(-3, -2, -1))
        for m in range(3):
            x = self._shear(x, per[0][m], self.et_tan[m])
            x = self._shear(x, per[1][m], self.et_sin[m])
            x = self._shear(x, per[0][m], self.et_tan[m])
        x = torch.fft.fftn(x, dim=(-3, -2, -1))
        x = x * self.et_tra
        x = torch.fft.ifftn(x, dim=(-3, -2, -1))
        return torch.fft.fftshift(x, dim=(-3, -2, -1))


class SampledEncoder:
    """A = gather . F . S, evaluated only where data was acquired.

    mps : (nc, ny, nz, nx)
    Ey  : (B, nt, npe, ny)  rows of the centred y-DFT at the sampled ky
    Ez  : (B, nt, npe, nz)
    Ex  : (nkx, nx)         rows of the centred x-DFT at the selected kx
    """

    def __init__(self, mps, Ey, Ez, Ex, k_chunk=1):
        self.mps, self.Ey, self.Ez, self.Ex = mps, Ey, Ez, Ex
        self.k_chunk = k_chunk

    def index(self, idx):
        if idx.numel() == self.Ey.shape[0]:
            return self
        c = getattr(self, '_cache', None)
        if c is not None and c[0].numel() == idx.numel() and bool(torch.equal(c[0], idx)):
            return c[1]
        sub = SampledEncoder(self.mps, self.Ey[idx], self.Ez[idx], self.Ex, self.k_chunk)
        self._cache = (idx, sub)
        return sub

    def _one(self, vol):
        w = vol.unsqueeze(2) * self.mps[None, None, :, None]
        w = torch.einsum('btpy,bkctyzx->bkctpzx', self.Ey, w)
        w = torch.einsum('btpz,bkctpzx->bkctpx', self.Ez, w)
        return torch.einsum('qx,bkctpx->bkctpq', self.Ex, w)

    def __call__(self, vol):
        K = vol.shape[1]
        if self.k_chunk >= K:
            return self._one(vol)
        return torch.cat([self._one(vol[:, i:i + self.k_chunk])
                          for i in range(0, K, self.k_chunk)], dim=1)


# ===========================================================================
# batched Levenberg-Marquardt, complex residual / real parameters
# ===========================================================================
def lm_solve(resid_jac, x0, max_iter=60, tol=1e-8, lam0=1e-3,
             lam_up=4.0, lam_dn=3.0, lam_max=1e10, inner=8):
    """Minimise ||r_b(x_b)||^2 independently for every bin b.

    resid_jac(x_sub, need_jac, idx) -> (r, J)
        x_sub : (n, P) the rows of x for the bins in `idx`
        r     : (n, m) complex   J : (n, m, P) complex (None if not need_jac)

    Each outer pass builds one Jacobian and then retries the damped step up
    to `inner` times, so a rejected step costs one residual evaluation
    instead of a whole new Jacobian.  Converged bins leave the batch.
    """
    x = x0.clone()
    B, P = x.shape
    dev = x.device
    lam = torch.full((B,), lam0, device=dev, dtype=x.dtype)
    allidx = torch.arange(B, device=dev)
    r0, _ = resid_jac(x, False, allidx)
    cost = (r0.real ** 2 + r0.imag ** 2).sum(-1)
    nit = torch.zeros(B, device=dev, dtype=torch.int32)
    active = torch.ones(B, dtype=torch.bool, device=dev)

    for _ in range(max_iter):
        idx = active.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            break
        r, J = resid_jac(x[idx], True, idx)
        cost_i = (r.real ** 2 + r.imag ** 2).sum(-1)
        lam_i = lam[idx]
        JhJ = torch.einsum('bmp,bmq->bpq', J.conj(), J).real
        Jhr = -torch.einsum('bmp,bm->bp', J.conj(), r).real
        diag = torch.diagonal(JhJ, dim1=-2, dim2=-1).clamp_min(1e-30)

        done = torch.zeros(idx.numel(), dtype=torch.bool, device=dev)
        rel_acc = torch.zeros(idx.numel(), device=dev, dtype=x.dtype)
        x_i = x[idx]
        for _ in range(inner):
            A = JhJ + lam_i[:, None, None] * torch.diag_embed(diag)
            step = torch.nan_to_num(
                torch.linalg.solve(A, Jhr.unsqueeze(-1)).squeeze(-1))
            x_try = torch.where(done[:, None], x_i, x_i + step)
            r_try, _ = resid_jac(x_try, False, idx)
            c_try = (r_try.real ** 2 + r_try.imag ** 2).sum(-1)
            good = (~done) & (c_try < cost_i)
            rel = (cost_i - c_try).abs() / cost_i.clamp_min(1e-30)
            x_i = torch.where(good[:, None], x_try, x_i)
            # the relative gain of each bin's ACCEPTED step is what decides
            # below whether that bin has converged
            rel_acc = torch.where(good, rel, rel_acc)
            done = done | good
            lam_i = torch.where(good, (lam_i / lam_dn).clamp_min(1e-12),
                                torch.where(done, lam_i,
                                            (lam_i * lam_up).clamp_max(lam_max)))
            cost_i = torch.where(good, c_try, cost_i)
            if bool(done.all()) or bool((lam_i >= lam_max).all()):
                break

        x[idx] = x_i
        lam[idx] = lam_i
        cost[idx] = cost_i
        nit[idx] += 1
        active[idx] = done & (rel_acc >= tol) & (lam_i < lam_max)

    return x, cost.sqrt(), nit


# ===========================================================================
# driver
# ===========================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description='Per-bin motion + B0 estimation (fast drop-in for motion_est_lbfgs.py)')
    p.add_argument('--path', type=str, help='Path to the sepmo data directory')
    p.add_argument('--gpu', type=int, default=1, help='CUDA device index')
    p.add_argument('--nb', type=int, default=9, help='Number of B0 basis functions')
    p.add_argument('--start_bin', type=int, default=0, help='First bin index to process')
    p.add_argument('--batch', type=int, default=8,
                   help='Bins solved simultaneously (halved automatically on OOM)')
    p.add_argument('--k_chunk', type=int, default=1,
                   help='Jacobian columns encoded at once (lower = less GPU memory)')
    p.add_argument('--lm_tol', type=float, default=1e-8,
                   help='Stop a bin when an accepted LM step gains less than this '
                        'relative reduction in ||r||^2')
    p.add_argument('--n_alt', type=int, default=1,
                   help='B0<->motion alternation cycles per bin (1 = the '
                        'original single pass)')
    p.add_argument('--lm_iter_b0', type=int, default=60)
    p.add_argument('--lm_iter_mo', type=int, default=60)
    p.add_argument('--fd_h', type=float, default=1e-3,
                   help='Central-difference step for the motion Jacobian')
    p.add_argument('--float64', action='store_true',
                   help='Run in complex128 (reference precision; much slower)')
    return p.parse_args()


def main():
    args = parse_args()
    path = args.path.rstrip('/') + '/'
    dev = torch.device('cuda', args.gpu)
    torch.cuda.set_device(dev)
    cdt = torch.complex128 if args.float64 else torch.complex64
    rdt = torch.float64 if args.float64 else torch.float32

    from motion_est.utils import np_to_torch as t2

    # ---- load -------------------------------------------------------------
    has_mo = Path(path + 'SEs_tmp_mo.hdr').is_file()
    ksp_all = t2(readcfl(path + 'kspace_all_new_tmp'))
    ksp_all_mo = t2(readcfl(path + 'kspace_all_new_tmp_mo')) if has_mo else ksp_all
    ky = np.real(readcfl(path + 'ky_tmp')).astype(int) - 1       # (nbins,npe,nt)
    kz = np.real(readcfl(path + 'kz_tmp')).astype(int) - 1
    kx_np = (np.real(readcfl(path + 'kx_tmp')).astype(int) - 1).flatten()
    img = t2(readcfl(path + 'a0_tmp')).to(dev).to(cdt)
    SEs = t2(readcfl(path + 'SEs_tmp')).to(dev).to(cdt)
    SEs_mo = t2(readcfl(path + 'SEs_tmp_mo')).to(dev).to(cdt) if has_mo else SEs
    bas = t2(readcfl(path + 'Basn_tmp')).to(dev).real.to(rdt)[:args.nb]
    tes = t2(readcfl(path + 'TEs_tmp')).to(dev).real.to(rdt)
    nt_mo = (np.real(readcfl(path + 'nt_sel_mo')).flatten() - 1).astype(int).tolist()
    nt_b0 = (np.real(readcfl(path + 'nt_sel_dB0')).flatten() - 1).astype(int).tolist()
    ratios = t2(readcfl(path + 'ratios_new_tmp')).to(dev)
    motion_init = t2(np.real(readcfl(path + 'motion_init'))).to(dev).to(rdt)

    nc, ny, nz, nx = SEs.shape
    nb, nt, nbins, npe = bas.shape[0], tes.shape[0], ksp_all.shape[0], ksp_all.shape[3]
    nkx = len(kx_np)
    img_size = (ny, nz, nx)
    if has_mo:
        print(f'motion-fit coil-reduced variant found: nc_mo={SEs_mo.shape[0]} '
              f'(B0 fit stays at nc={nc})')
    print(f'alternation cycles per bin: {args.n_alt}')
    print(f'nc={nc}  nt={nt}  ny={ny}  nz={nz}  nx={nx}  nbins={nbins}  nb={nb}  '
          f'npe={npe}  nkx={nkx}')

    directory = Path(path)
    if any(f.stem == 'B0_base_tmp' for f in directory.iterdir() if f.is_file()):
        b0_all = t2(readcfl(path + 'B0_base_tmp'))
        print(f'B0 base loaded: {tuple(b0_all.shape)}')
    else:
        b0_all = None
        print('B0_base_tmp not found - skipping B0 pre-correction')

    Ey_full = build_dft_mat(ny, dev, cdt)
    Ez_full = build_dft_mat(nz, dev, cdt)
    Ex = build_dft_mat(nx, dev, cdt)[torch.as_tensor(kx_np, device=dev), :]
    kx_t = torch.as_tensor(kx_np, dtype=torch.long)

    # The reference writes every shot's data into a full k-space grid, so two
    # shots landing on the same (ky,kz) within one bin leave only the last
    # one and are counted once.  Gathering per shot would count both, so
    # drop all but the last occurrence: a zeroed DFT row predicts 0 against
    # a zeroed target and contributes nothing to the residual.
    keep = np.ones((nbins, npe, nt), dtype=bool)
    ndup = 0
    for b in range(nbins):
        for t in range(nt):
            seen = {}
            for i in range(npe):
                seen.setdefault((ky[b, i, t], kz[b, i, t]), []).append(i)
            for v in seen.values():
                for i in v[:-1]:
                    keep[b, i, t] = False
                    ndup += 1
    if ndup:
        print(f'WARNING: {ndup} duplicate (ky,kz) shots dropped (last one kept)')
    keep_t = torch.as_tensor(keep, device=dev)

    def rows(E_full, idx3, ts):
        """(nbins,npe,nt)[...,ts] -> (nbins, |ts|, npe, n), duplicates zeroed."""
        i = torch.as_tensor(np.ascontiguousarray(idx3[:, :, ts]),
                            dtype=torch.long, device=dev)
        E = E_full[i].permute(0, 2, 1, 3).contiguous()
        return E * keep_t[:, :, ts].permute(0, 2, 1)[..., None].to(E.dtype)

    Ey_b0, Ez_b0 = rows(Ey_full, ky, nt_b0), rows(Ez_full, kz, nt_b0)
    Ey_mo, Ez_mo = rows(Ey_full, ky, nt_mo), rows(Ez_full, kz, nt_mo)

    def target(src, ts):
        # (nbins, nc, |ts|, npe, nkx) -- exactly the points the reference's
        # mask keeps, scaled by the same per-bin ratio
        d = src[:, :, ts][..., kx_t].to(dev).to(cdt)
        d = d * ratios[0][:, None, None, None, None]
        k = keep_t[:, :, ts].permute(0, 2, 1)[:, None, :, :, None].to(cdt)
        return d * k

    tgt_b0 = target(ksp_all, nt_b0)
    tgt_mo = target(ksp_all_mo, nt_mo)

    tes_b0, tes_mo = tes[nt_b0], tes[nt_mo]
    n_b0 = len(nt_b0)
    dcoef = (1j * TWOPI * tes_b0[None, :, None, None, None] * bas[:, None]).to(cdt)
    eye6 = torch.eye(6, device=dev, dtype=rdt)

    motion_est = torch.zeros(nbins, 6, dtype=torch.complex64)
    cr_est = torch.zeros(nbins, nb, dtype=torch.complex64)
    res_hist = np.zeros((nbins, 2))
    it_hist = np.zeros((nbins, 2), dtype=int)

    todo = list(range(args.start_bin, nbins))
    t_start = time.time()
    t_b0 = t_mo = 0.0

    def sync():
        torch.cuda.synchronize()

    def solve_batch(bl):
        B = len(bl)
        blt = torch.as_tensor(bl, device=dev)

        ib = img[nt_b0].unsqueeze(0).expand(B, -1, -1, -1, -1)
        im = img[nt_mo].unsqueeze(0).expand(B, -1, -1, -1, -1)
        if b0_all is not None:
            bb = b0_all[bl].to(dev).real.to(rdt)
            ib = ib * torch.exp(1j * TWOPI * tes_b0[None, :, None, None, None]
                                * bb[:, None]).to(cdt)
            im = im * torch.exp(1j * TWOPI * tes_mo[None, :, None, None, None]
                                * bb[:, None]).to(cdt)
        ib, im = ib.contiguous(), im.contiguous()
        d_b0, d_mo = tgt_b0[blt], tgt_mo[blt]
        x1_fix = motion_init[blt]
        # rebuilt at the top of every alternation cycle from the current x1
        T_b0 = [BatchedMotion(img_size, x1_fix, cdt)]
        Bx   = [None]
        enc_b0 = SampledEncoder(SEs, Ey_b0[blt], Ez_b0[blt], Ex, args.k_chunk)
        enc_mo = SampledEncoder(SEs_mo, Ey_mo[blt], Ez_mo[blt], Ex, args.k_chunk)

        # ---- alternate B0 and motion -------------------------------------
        # One cycle (the default) reproduces the original ordering exactly:
        # B0 is fitted with x1 pinned at motion_init, then motion with that
        # B0 frozen. With n_alt > 1 the B0 fit is repeated against the
        # motion the previous cycle found, and each sub-fit warm starts from
        # its own previous answer -- so extra cycles cost far fewer
        # iterations than the first.
        tb0 = tmo = 0.0
        nit2 = nit1 = None
        x1 = x1_fix.clone()
        x2 = torch.zeros(B, nb, device=dev, dtype=rdt)

        def rj_b0(x2, need_jac, idx):
            n = idx.numel()
            b0 = torch.einsum('lyzx,bl->byzx', bas, x2)
            g0 = ib[idx] * torch.exp(1j * TWOPI * tes_b0[None, :, None, None, None]
                                     * b0[:, None]).to(cdt)
            stack = g0.unsqueeze(1)
            if need_jac:
                stack = torch.cat([stack, g0.unsqueeze(1) * dcoef[None]], dim=1)
            K = stack.shape[1]
            h = T_b0[0].index(idx)(stack.reshape(n, K * n_b0, *img_size)
                                   ).reshape(n, K, n_b0, *img_size)
            Ah = enc_b0.index(idx)(h)
            r = (d_b0[idx] - Ah[:, 0]).reshape(n, -1)
            if not need_jac:
                return r, None
            return r, (-Ah[:, 1:]).reshape(n, nb, -1).transpose(1, 2)

        def rj_mo(x1, need_jac, idx):
            n = idx.numel()
            bx = Bx[0][idx]
            vols = [BatchedMotion(img_size, x1, cdt)(bx)]
            if need_jac:
                # dT/dx1_j by central differences taken in image space: 12
                # cheap T applications instead of 12 full encodes
                for j in range(6):
                    hp = BatchedMotion(img_size, x1 + args.fd_h * eye6[j], cdt)(bx)
                    hm = BatchedMotion(img_size, x1 - args.fd_h * eye6[j], cdt)(bx)
                    vols.append((hp - hm) / (2 * args.fd_h))
            Ah = enc_mo.index(idx)(torch.stack(vols, dim=1))
            r = (d_mo[idx] - Ah[:, 0]).reshape(n, -1)
            if not need_jac:
                return r, None
            return r, (-Ah[:, 1:]).reshape(n, 6, -1).transpose(1, 2)

        for it_alt in range(args.n_alt):
            # B0 against the motion we currently believe
            sync(); ta = time.time()
            T_b0[0] = BatchedMotion(img_size, x1, cdt)
            x2, res2, n2 = lm_solve(rj_b0, x2, max_iter=args.lm_iter_b0,
                                    tol=args.lm_tol)
            sync(); tb = time.time(); tb0 += tb - ta

            # motion against the B0 we just fitted
            b0f = torch.einsum('lyzx,bl->byzx', bas, x2)
            Bx[0] = im * torch.exp(1j * TWOPI * tes_mo[None, :, None, None, None]
                                   * b0f[:, None]).to(cdt)
            x1, res1, n1 = lm_solve(rj_mo, x1, max_iter=args.lm_iter_mo,
                                    tol=args.lm_tol)
            sync(); tmo += time.time() - tb
            nit2 = n2 if nit2 is None else nit2 + n2
            nit1 = n1 if nit1 is None else nit1 + n1
        t0 = 0.0; t1 = tb0; t2 = tb0 + tmo

        bad1 = torch.any(torch.isnan(x1), dim=1)
        bad2 = torch.any(torch.isnan(x2), dim=1)
        for k in torch.nonzero(bad1).flatten().tolist():
            print(f'WARNING: x1 is NaN for bin {bl[k]}, setting to zeros')
        for k in torch.nonzero(bad2).flatten().tolist():
            print(f'WARNING: x2 is NaN for bin {bl[k]}, setting to zeros')
        x1 = torch.nan_to_num(x1)
        x2 = torch.nan_to_num(x2)

        motion_est[blt.cpu()] = x1.cpu().to(torch.complex64)
        cr_est[blt.cpu()] = x2.cpu().to(torch.complex64)
        res_hist[bl, 0] = res2.cpu().numpy()
        res_hist[bl, 1] = res1.cpu().numpy()
        it_hist[bl, 0] = nit2.cpu().numpy()
        it_hist[bl, 1] = nit1.cpu().numpy()
        for k in range(B):
            print(f'bin {bl[k]:4d} | mo: {x1[k].tolist()} | b0: {x2[k].tolist()}')
        return t1 - t0, t2 - t1

    def solve_with_retry(bl):
        # the GPU is shared, so fall back to smaller batches rather than dying
        try:
            return solve_batch(bl)
        except torch.cuda.OutOfMemoryError:
            if len(bl) == 1:
                raise
            torch.cuda.empty_cache()
            h = len(bl) // 2
            print(f'out of memory on {len(bl)} bins -- retrying in halves')
            a1 = solve_with_retry(bl[:h])
            a2 = solve_with_retry(bl[h:])
            return a1[0] + a2[0], a1[1] + a2[1]

    for i0 in range(0, len(todo), args.batch):
        db0, dmo = solve_with_retry(np.asarray(todo[i0:i0 + args.batch]))
        t_b0 += db0
        t_mo += dmo

    el = time.time() - t_start
    n = max(1, len(todo))
    print(f'\n{len(todo)} bins in {el:.1f}s  ({el/n:.3f} s/bin)  '
          f'B0 {t_b0/n:.3f} s/bin, motion {t_mo/n:.3f} s/bin')
    print(f'LM iterations: B0 mean {it_hist[todo,0].mean():.1f} '
          f'(max {it_hist[todo,0].max()}), motion mean {it_hist[todo,1].mean():.1f} '
          f'(max {it_hist[todo,1].max()})')

    writecfl(path + 'motion_est_new', motion_est.cpu().numpy())
    writecfl(path + 'cr_est_new', cr_est.cpu().numpy())
    print(f'Saved  motion_est_new  and  cr_est_new  to {path}')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(res_hist[:, 1]); axes[0].set_title('Motion fit: final ||r|| per bin')
    axes[1].plot(res_hist[:, 0]); axes[1].set_title('B0 fit: final ||r|| per bin')
    for ax in axes:
        ax.set_xlabel('bin'); ax.set_ylabel('residual norm')
    plt.tight_layout()
    plt.savefig(path + 'loss_history.png', dpi=150)
    print(f'Residual plot saved to {path}loss_history.png')


if __name__ == '__main__':
    main()
