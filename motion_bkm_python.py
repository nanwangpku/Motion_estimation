"""Pure-Python port of motion_bkm_main.m's python-estimation path (former "option 2").

Loads the raw SMENA .mat directly (no MATLAB, no .cfl round-trip), reproduces the
MATLAB preprocessing in numpy/torch, and calls motion_est_lbfgs.estimate() for the
per-bin LBFGS motion + B0 fit.
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'python' / 'motion_est-main'))

import motion_est_lbfgs  # noqa: E402
from motion_est.linops_nw import fftc  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description='Pure-Python SMENA motion/B0 estimation')
    parser.add_argument('--mat_path', type=str, default=str(HERE / 'data' / 'example_smena_data_meGRE2.mat'))
    parser.add_argument('--gpu', type=int, default=None,
                         help='CUDA device index (default: auto-pick the most-free GPU)')
    parser.add_argument('--nb', type=int, default=9)
    parser.add_argument('--start_bin', type=int, default=0)
    parser.add_argument('--out', type=str, default=None,
                         help='Output .mat path (default: data/motion_estimation.mat next to mat_path, '
                              'matching motion_bkm_main.m so recon_megre.m can load either one)')
    return parser.parse_args()


def choose_gpu():
    """Mirrors choose_GPU.m: pick the device with the highest free/total memory ratio."""
    if not torch.cuda.is_available():
        raise RuntimeError('No CUDA device available')
    best, best_frac = 0, -1.0
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        frac = free / total
        if frac > best_frac:
            best, best_frac = i, frac
    return best


def mat_t(x):
    """Full axis reversal: converts an h5py-loaded MATLAB v7.3 array (HDF5-native
    axis order) back to MATLAB's own indexing convention (size()-compatible)."""
    return np.asarray(x).T


def to_complex(x):
    """MATLAB/HDF5 stores purely-real arrays as plain float (no 'real'/'imag' fields) —
    a handful of kspace_cart_orig_fid blocks hit this (imaginary part optimized away
    as exactly zero), so handle both the structured-complex and plain-real cases."""
    if x.dtype.names:
        return x['real'].astype(np.complex64) + 1j * x['imag'].astype(np.complex64)
    return x.astype(np.complex64)


class KspaceCells:
    """Lazily dereferences data_str.kspace_cart_orig_fid{ii_out}{ii_in} (MATLAB's
    1-indexed cell-of-cells), returning MATLAB-shape (Nx, Nt, Ncoils) complex64 blocks
    for 0-indexed (ii_out0, ii_in0)."""

    def __init__(self, h5file, outer_refs):
        self.f = h5file
        self.outer_refs = outer_refs  # (n_outer, 1) object dataset of h5py refs

    def get(self, ii_out0, ii_in0):
        inner_refs = self.f[self.outer_refs[ii_out0, 0]]
        block = self.f[inner_refs[ii_in0, 0]]
        return mat_t(to_complex(np.asarray(block)))


def load_mat(mat_path):
    f = h5py.File(mat_path, 'r')
    ds = f['data_str']

    data = dict(
        a_nav_refs  = mat_t(to_complex(np.asarray(ds['a_nav_refs']))),   # (Ny,Nz,Nx,1,Nt)
        Bas         = mat_t(np.asarray(ds['Bas'])),                      # (Ny*Nz*Nx, 9)
        SEs_perm    = mat_t(to_complex(np.asarray(ds['SEs_perm']))),     # (Ny,Nz,Nx,Ncoils)
        TEs_fid     = mat_t(np.asarray(ds['TEs_fid'])),                  # (Nt, 1)
        ky_fid      = mat_t(np.asarray(ds['ky_fid'])),                   # (shot_fid_in, Nt)
        kz_fid      = mat_t(np.asarray(ds['kz_fid'])),
        ratios_all  = mat_t(np.asarray(ds['ratios_all'])).flatten(),     # (nshot_total,)
        size_nav    = mat_t(np.asarray(ds['size_nav'])).flatten(),       # (3,)
        voxel_size  = mat_t(np.asarray(ds['voxel_size'])).flatten(),     # (3,)
        flip_flag   = bool(np.asarray(ds['flip_flag']).flatten()[0]),
        shot_fid    = int(np.asarray(ds['shot_fid']).flatten()[0]),
        shot_fid_in = int(np.asarray(ds['shot_fid_in']).flatten()[0]),
        nshot_total = int(np.asarray(ds['nshot_total']).flatten()[0]),
    )
    data['kcells'] = KspaceCells(f, ds['kspace_cart_orig_fid'])
    data['h5file'] = f
    return data


def preprocess(data, fid_sel=4, fid_dis=20,
               nt_sel=np.arange(1, 7), nt_sel_dB0=np.array([1, 5]), nt_sel_mo=np.array([1, 3])):
    """Direct translation of motion_bkm_main.m's former "option 2" preprocessing block
    (kx_sel/Ncoils/Bas_norm setup + the per-bin kspace_tmp/ky_tmp/kz_tmp/ratios_sep loop).
    Coordinate VALUES (ky/kz pixel indices, kx range, nt_sel*) are kept 1-indexed,
    matching exactly what MATLAB's writecfl would have produced — only the Python-side
    loop bookkeeping is 0-indexed. This lets the result feed motion_est_lbfgs.estimate()
    completely unmodified.
    """
    a_nav_refs  = data['a_nav_refs']
    Bas         = data['Bas']
    SEs_perm    = data['SEs_perm']
    TEs_fid     = data['TEs_fid']
    ky_fid      = data['ky_fid']
    kz_fid      = data['kz_fid']
    ratios_all  = data['ratios_all']
    size_nav    = data['size_nav']
    flip_flag   = data['flip_flag']
    shot_fid    = data['shot_fid']
    nshot_total = data['nshot_total']
    kcells      = data['kcells']

    Ny_nav, Nz_nav, Nx_nav = size_nav.astype(int)
    kx_sel_1based = np.arange(int(np.ceil(Nx_nav / 5 + 1)), int(np.ceil(Nx_nav / 5 * 4)) + 1)
    Ncoils = SEs_perm.shape[3]

    period = fid_sel + fid_dis
    nbins_sep = int(np.ceil(nshot_total / period))

    nt_sel0 = nt_sel - 1
    Nt_sel = len(nt_sel)

    ky_sel = ky_fid[:, nt_sel0]
    kz_sel = kz_fid[:, nt_sel0]
    if flip_flag:
        ky_sel = Ny_nav + 1 - ky_sel
        kz_sel = Nz_nav + 1 - kz_sel

    kspace_tmp = np.zeros((nbins_sep, Ncoils, Nt_sel, period, Nx_nav), dtype=np.complex64)
    ky_tmp     = np.zeros((nbins_sep, period, Nt_sel), dtype=np.float64)
    kz_tmp     = np.zeros((nbins_sep, period, Nt_sel), dtype=np.float64)
    ratios_sep = np.zeros(nbins_sep, dtype=np.float64)

    resp_bin_sep0  = np.arange(nshot_total) // period
    fid_idx_1based = (np.arange(nshot_total) % shot_fid) + 1

    for ii_bin0 in range(nbins_sep):
        ii_shot0 = np.where(resp_bin_sep0 == ii_bin0)[0]
        for ii0, ii_all0 in enumerate(ii_shot0):
            ii_out0 = int(ii_all0 // shot_fid)
            ii_in0  = int(ii_all0 % shot_fid)

            # overwritten every iteration on purpose — replicates MATLAB's
            # ratios_sep(ii_bin) = ratios_all(ii_all), which is *not* indexed by ii,
            # so only the last shot in each block ends up surviving
            ratios_sep[ii_bin0] = ratios_all[ii_all0]

            block = kcells.get(ii_out0, ii_in0)          # (Nx, Nt_full, Ncoils)
            block_sel = block[:, nt_sel0, :]              # (Nx, Nt_sel, Ncoils)
            block_fft = fftc(torch.from_numpy(block_sel), dim=(0,))
            block_perm = block_fft.permute(2, 1, 0).numpy()   # (Ncoils, Nt_sel, Nx)

            kspace_tmp[ii_bin0, :, :, ii0, :] = block_perm

            row = int(fid_idx_1based[ii_all0]) - 1
            ky_tmp[ii_bin0, ii0, :] = ky_sel[row, :]
            kz_tmp[ii_bin0, ii0, :] = kz_sel[row, :]

    a0_tmp = np.transpose(a_nav_refs[:, :, :, 0:1, nt_sel0], (4, 0, 1, 2, 3))[..., 0]  # (Nt_sel,Ny,Nz,Nx)

    TEs_tmp = TEs_fid[nt_sel0, 0]  # (Nt_sel,)

    SEs_tmp = np.transpose(SEs_perm, (3, 0, 1, 2))  # (Ncoils,Ny,Nz,Nx)

    Bas_tmp = Bas.reshape(Ny_nav, Nz_nav, Nx_nav, 9, order='F')
    Bas_tmp = np.transpose(Bas_tmp, (3, 0, 1, 2))  # (9,Ny,Nz,Nx)
    weights = np.array([1, 10, 10, 10, 100, 100, 100, 100, 100], dtype=Bas_tmp.dtype).reshape(9, 1, 1, 1)
    Basn_tmp = Bas_tmp * weights

    raw = dict(
        kspace_all_new_tmp = kspace_tmp,
        ky_tmp              = ky_tmp,
        kz_tmp               = kz_tmp,
        kx_tmp               = kx_sel_1based.reshape(1, -1).astype(np.float64),
        a0_tmp               = a0_tmp,
        SEs_tmp              = SEs_tmp,
        Basn_tmp             = Basn_tmp,
        TEs_tmp              = TEs_tmp,
        nt_sel_mo            = nt_sel_mo.reshape(1, -1).astype(np.float64),
        nt_sel_dB0           = nt_sel_dB0.reshape(1, -1).astype(np.float64),
        ratios_new_tmp       = ratios_sep.reshape(1, -1),
        motion_init          = None,
        B0_base_tmp          = None,
    )
    meta = dict(fid_sel=fid_sel, fid_dis=fid_dis, resp_bin_sep=resp_bin_sep0 + 1,
                fid_idx=fid_idx_1based, nbins_sep=nbins_sep)
    return raw, meta


def main():
    args = parse_args()
    dev = torch.device(args.gpu if args.gpu is not None else choose_gpu())
    print(f'Using device: {dev}')

    data = load_mat(args.mat_path)
    raw, meta = preprocess(data)
    data['h5file'].close()

    motion_est, cr_est, loss_history1, loss_history2 = motion_est_lbfgs.estimate(
        raw, dev, nb_use=args.nb, start_bin=args.start_bin)

    out_path = args.out or str(Path(args.mat_path).parent / 'motion_estimation.mat')
    sio.savemat(out_path, dict(
        motion_est=motion_est.real.cpu().numpy(),
        cr_est=cr_est.real.cpu().numpy(),
        Bas=data['Bas'],
        Bas_norm=data['Bas'] * np.array([1, 10, 10, 10, 100, 100, 100, 100, 100]),
        voxel_size=data['voxel_size'],
        size_nav=data['size_nav'],
        nshot_total=data['nshot_total'],
        ratios_sep=raw['ratios_new_tmp'],
        TEs_fid=data['TEs_fid'],
        fid_idx=meta['fid_idx'],
        fid_sel=meta['fid_sel'],
        fid_dis=meta['fid_dis'],
        resp_bin_sep=meta['resp_bin_sep'],
    ))
    print(f'Saved results to {out_path}')


if __name__ == '__main__':
    main()
