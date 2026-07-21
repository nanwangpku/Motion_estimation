import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from motion_est.linops_nw import fftc, ifftc
from motion_est.imperfections.motion_full import motion_ops
from einops import einsum
from util import readcfl, writecfl
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Per-bin motion + B0 estimation with LBFGS')
    parser.add_argument('--path', type=str,
                        help='Path to the sepbatch data directory')
    parser.add_argument('--gpu', type=int, default=1,
                        help='CUDA device index (default: 1)')
    parser.add_argument('--nb', type=int, default=9,
                        help='Number of B0 basis functions to use (default: 9)')
    parser.add_argument('--start_bin', type=int, default=0,
                        help='First bin index to process (default: 0, useful for resuming)')
    return parser.parse_args()


def forward(x1, x2, img, mps, mask, bas, tes, ksp, img_size, pad_num):
    b0  = einsum(bas, x2.squeeze(0), 'nl ny nz nx, nl -> ny nz nx')
    phs = torch.exp(1j * 2 * torch.pi * einsum(b0, tes, 'ny nz nx, nt -> nt ny nz nx'))
    Bx  = img * phs

    T, _ = motion_ops(img_size, x1.squeeze(0), pad_num)
    TBx  = T(Bx)

    STBx   = einsum(TBx, mps,  'nt nro npe nx, nc nro npe nx -> nc nt nro npe nx')
    FSTBx  = fftc(STBx, dim=(-3, -2, -1))
    MFSTBx = einsum(FSTBx, mask, 'nc nt nro npe nx, nt nro npe nx -> nc nt nro npe nx')

    return torch.norm(ksp - MFSTBx) / torch.norm(ksp)


def load_from_cfl(path):
    """Read the sepbatch data directory written by MATLAB's writecfl into a raw dict.

    Values are kept in their on-disk convention (e.g. 1-indexed ky/kz coordinates) —
    all normalization happens in estimate(), so any other loader (e.g. a pure-Python
    preprocessing pipeline) can build the same dict without touching estimate().
    """
    path = path.rstrip('/') + '/'
    raw = dict(
        kspace_all_new_tmp = readcfl(path + 'kspace_all_new_tmp'),
        ky_tmp              = readcfl(path + 'ky_tmp'),
        kz_tmp               = readcfl(path + 'kz_tmp'),
        kx_tmp               = readcfl(path + 'kx_tmp'),
        a0_tmp               = readcfl(path + 'a0_tmp'),
        SEs_tmp              = readcfl(path + 'SEs_tmp'),
        Basn_tmp             = readcfl(path + 'Basn_tmp'),
        TEs_tmp              = readcfl(path + 'TEs_tmp'),
        nt_sel_mo            = readcfl(path + 'nt_sel_mo'),
        nt_sel_dB0           = readcfl(path + 'nt_sel_dB0'),
        ratios_new_tmp       = readcfl(path + 'ratios_new_tmp'),
    )

    raw['motion_init'] = readcfl(path + 'motion_init') if Path(path + 'motion_init.cfl').exists() else None

    directory = Path(path)
    if any(f.stem == 'B0_base_tmp' for f in directory.iterdir() if f.is_file()):
        raw['B0_base_tmp'] = readcfl(str(directory / 'B0_base_tmp'))
    else:
        raw['B0_base_tmp'] = None

    return raw


def estimate(raw, dev, nb_use=9, start_bin=0):
    # ── Data normalization ────────────────────────────────────────────────────
    def to_torch(arr):
        from motion_est.utils import np_to_torch
        return np_to_torch(arr)

    ksp_all   = to_torch(raw['kspace_all_new_tmp'])
    ky_sel    = (raw['ky_tmp'] - 1).astype(int)
    kz_sel    = (raw['kz_tmp'] - 1).astype(int)
    kx_sel_np = (raw['kx_tmp'] - 1).astype(int)
    kx_sel    = slice(kx_sel_np[0, 0] - 1, kx_sel_np[0, -1] - 1)

    img       = to_torch(raw['a0_tmp']).to(dev)
    SEs       = to_torch(raw['SEs_tmp']).to(dev)
    bas       = to_torch(raw['Basn_tmp']).to(dev).real.double()
    tes       = to_torch(raw['TEs_tmp']).to(dev).real.double()
    nt_mo     = (np.real(raw['nt_sel_mo']).flatten() - 1).astype(int).tolist()
    nt_b0     = (np.real(raw['nt_sel_dB0']).flatten() - 1).astype(int).tolist()
    ratios_gt = to_torch(raw['ratios_new_tmp']).to(dev)
    if raw['motion_init'] is not None:
        motion_init = to_torch(np.real(raw['motion_init'])).to(dev)
    else:
        motion_init = torch.zeros(ksp_all.shape[0], 6, dtype=torch.float64, device=dev)
        print('motion_init not found — initializing to zeros')

    bas = bas[:nb_use, ...]

    nc, ny, nz, nx = SEs.shape
    nb    = bas.shape[0]
    nt    = tes.shape[0]
    nbins = ksp_all.shape[0]

    print(f'nc={nc}  nt={nt}  ny={ny}  nz={nz}  nx={nx}  nbins={nbins}  nb={nb}')

    if raw['B0_base_tmp'] is not None:
        b0_all = to_torch(raw['B0_base_tmp'])
        print(f'B0 base loaded: {b0_all.shape}')
    else:
        b0_all = None
        print('B0_base_tmp not found — skipping B0 pre-correction')

    img_size = torch.tensor([ny, nz, nx], device=dev, dtype=torch.int)
    pad_num  = torch.tensor([0,  0,  0],  device=dev, dtype=torch.int)

    # ── Result buffers ────────────────────────────────────────────────────────
    loss_history1 = [[] for _ in range(nbins)]
    loss_history2 = [[] for _ in range(nbins)]
    motion_est    = torch.zeros(nbins, 6, dtype=torch.complex64)
    cr_est        = torch.zeros(nbins, nb, dtype=torch.complex64)
    ksp           = torch.zeros((nc, nt, ny, nz, nx), dtype=torch.complex64).to(dev)

    # ── Per-bin LBFGS optimization ────────────────────────────────────────────
    for ii_bin in range(start_bin, nbins):
        mask = torch.zeros(nt, ny, nz, nx, dtype=torch.float32)
        for ii_t in range(nt):
            ksp[:, ii_t, ky_sel[ii_bin, :, ii_t], kz_sel[ii_bin, :, ii_t], :] = \
                ksp_all[ii_bin, :, ii_t, ...].to(dev)
            mask[ii_t, ky_sel[ii_bin, :, ii_t], kz_sel[ii_bin, :, ii_t], kx_sel] = True
        mask = mask.to(dev)
        ksp  = ksp * mask * ratios_gt[0, ii_bin]

        x1_fix = torch.tensor(motion_init[ii_bin, :], device=dev, dtype=torch.float64)

        # motion subset (echo times 0 and 1)
        # nt_mo       = [0, 1]
        tes_mo      = tes[nt_mo]
        ksp_mo      = ksp[:, nt_mo, ...]
        mask_mo     = mask[nt_mo, ...]
        img_base_mo = img[nt_mo, ...]

        # B0 subset (echo times 0 and 2)
        # nt_b0       = [0, 2]
        tes_b0      = tes[nt_b0]
        ksp_b0      = ksp[:, nt_b0, ...]
        mask_b0     = mask[nt_b0, ...]
        img_base_b0 = img[nt_b0, ...]
        if b0_all is not None:
            phs_base_b0 = torch.exp(1j * 2 * torch.pi * einsum(
                b0_all[ii_bin, ...].to(dev), tes_b0, 'ny nz nx, nt -> nt ny nz nx'))
            img_base_b0 = img_base_b0 * phs_base_b0
            phs_base_mo = torch.exp(1j * 2 * torch.pi * einsum(
                b0_all[ii_bin, ...].to(dev), tes_mo, 'ny nz nx, nt -> nt ny nz nx'))
            img_base_mo = img_base_mo * phs_base_mo

        x1 = torch.tensor(motion_init[ii_bin, :], device=dev,
                          dtype=torch.float64, requires_grad=True)
        x2 = torch.zeros(nb, device=dev, dtype=torch.float64, requires_grad=True)

        optimizer1 = torch.optim.LBFGS([x1], lr=1e-1, max_iter=200,
                                        tolerance_grad=1e-7, tolerance_change=1e-9,
                                        history_size=20)
        optimizer2 = torch.optim.LBFGS([x2], lr=5e-2, max_iter=1000,
                                        tolerance_grad=1e-7, tolerance_change=1e-9,
                                        history_size=20)

        def closure2():
            optimizer2.zero_grad()
            loss = forward(x1_fix, x2, img_base_b0, SEs, mask_b0, bas, tes_b0, ksp_b0,
                           img_size, pad_num)
            loss.backward()
            loss_history2[ii_bin].append(loss.item())
            return loss

        optimizer2.step(closure2)
        x2_fix = x2.detach().clone()

        def closure1():
            optimizer1.zero_grad()
            loss = forward(x1, x2_fix, img_base_mo, SEs, mask_mo, bas, tes_mo, ksp_mo,
                           img_size, pad_num)
            loss.backward()
            loss_history1[ii_bin].append(loss.item())
            return loss

        optimizer1.step(closure1)
        x1_fix = x1.detach().clone()

        if torch.any(torch.isnan(x1_fix)):
            print(f'WARNING: x1 is NaN for bin {ii_bin}, setting to zeros')
            x1_fix = torch.zeros_like(x1_fix)
        if torch.any(torch.isnan(x2_fix)):
            print(f'WARNING: x2 is NaN for bin {ii_bin}, setting to zeros')
            x2_fix = torch.zeros_like(x2_fix)

        print(f'bin {ii_bin:4d} | mo: {x1_fix.tolist()} | b0: {x2_fix.tolist()}')
        motion_est[ii_bin, :] = x1_fix
        cr_est[ii_bin, :]     = x2_fix

    return motion_est, cr_est, loss_history1, loss_history2


def main():
    args = parse_args()
    path = args.path.rstrip('/') + '/'
    print(f'path{path}')
    dev  = torch.device(args.gpu)

    raw = load_from_cfl(path)
    motion_est, cr_est, loss_history1, loss_history2 = estimate(
        raw, dev, nb_use=args.nb, start_bin=args.start_bin)
    nbins = motion_est.shape[0]

    # ── Save ──────────────────────────────────────────────────────────────────
    writecfl(path + 'motion_est_new', motion_est.cpu().numpy())
    writecfl(path + 'cr_est_new',     cr_est.cpu().numpy())
    print(f'Saved  motion_est_new  and  cr_est_new  to {path}')

    # ── Loss plots ────────────────────────────────────────────────────────────
    plot_bin = min(4, nbins - 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(loss_history1[plot_bin])
    axes[0].set_title(f'Motion loss (x1) — bin {plot_bin}')
    axes[0].set_xlabel('LBFGS closure calls')
    axes[0].set_ylabel('Relative residual')

    axes[1].plot(loss_history2[plot_bin])
    axes[1].set_title(f'B0 loss (x2) — bin {plot_bin}')
    axes[1].set_xlabel('LBFGS closure calls')
    axes[1].set_ylabel('Relative residual')

    plt.tight_layout()
    fig_path = path + 'loss_history.png'
    plt.savefig(fig_path, dpi=150)
    print(f'Loss plot saved to {fig_path}')


if __name__ == '__main__':
    main()
