import gc
from textwrap import indent
from pyparsing import Opt
import torch
import torch.nn as nn
import numpy as np

from dataclasses import dataclass
from motion_est.fourier import fft, ifft
from motion_est.utils import batch_iterator, gen_grd
from motion_est.pad import PadLast
from motion_est.imperfections.imperfection import imperfection
from motion_est.linops import subspace_linop, batching_params
from motion_est.linops import linop
from motion_est.fourier import (
    gridded_nufft,
    sigpy_nufft,
    torchkb_nufft,
    NUFFT
)
from motion_est.multi_coil.grappa_est import train_kernels
from motion_est.indexing import multi_grid
from einops import rearrange, einsum
from typing import Optional
from tqdm import tqdm
from functools import reduce
import operator

# NW, 20241220, new reordering by loop over phase
class subspace_phase_linop(linop):

    def __init__(self,
                 im_size: tuple,
                 trj: torch.Tensor,
                 mps: torch.Tensor,
                 phi: torch.Tensor,
                 motion_bin: torch.Tensor,
                 dcf: Optional[torch.Tensor] = None,
                 nufft: Optional[NUFFT] = None,
                 imperf_model: Optional[tuple] = None,
                 imperf_model_motion: Optional[imperfection] = None,
                 use_toeplitz: Optional[bool] = False,
                 bparams: Optional[batching_params] = batching_params()):
        """
        Parameters
        ----------
        im_size : tuple 
            image dims as tuple of ints (dim1, dim2, ...)
        trj : torch.tensor <float> | GPU
            The k-space trajectory with shape (npe, ntr, nro, d). 
                we assume that trj values are in [-n/2, n/2] (for nxn grid)
        mps : torch.tensor <complex> | GPU
            sensititvity maps with shape (ncoil, ndim1, ..., ndimN)
        phi : torch.tensor <complex> | GPU
            subspace basis with shape (nsub, ntr)
        motion_bin : torch.tensor <int> | GPU
            which pe belongs to which motion states (1,npe)
        dcf : torch.tensor <float> | GPU
            the density comp. functon with shape (nro, ...)
        nufft : NUFFT
            the nufft object, defaults to torchkbnufft
        imperf_model : a struct cattying B0 and TEs
            
        imperf_model_motion : motion operator
            
        use_toeplitz : bool
            toggles toeplitz normal operator
        bparams : batching_params
            contains the batch sizes for the coils, subspace coeffs, and field segments
        """
        
        ishape = (phi.shape[0], *im_size)
        oshape = (mps.shape[0], *trj.shape[:-1])
        super().__init__(ishape, oshape)

        # Consts
        torch_dev = trj.device
        assert phi.device == torch_dev
        assert mps.device == torch_dev

        # Default params
        if nufft is None:
            nufft = torchkb_nufft(im_size, torch_dev.index)
        if dcf is None:
            dcf = torch.ones(trj.shape[:-1], dtype=torch.float32, device=torch_dev)
        else:
            assert dcf.device == torch_dev

        if imperf_model is None:
            b0 = torch.zeros((1,*mps.shape), dtype=torch.float32, device=torch_dev)
            tes = torch.zeros((phi.shape[1]), dtype=torch.float32, device=torch_dev)
        else:
            b0 = imperf_model[0]
            tes = imperf_model[1]
            
        # Rescale and type cast
        trj = nufft.rescale_trajectory(trj).type(torch.float32)
        dcf = dcf.type(torch.float32)
        mps = mps.type(torch.complex64)
        phi = phi.type(torch.complex64)
        b0 = b0.type(torch.complex64)
        tes = tes.type(torch.complex64)
        motion_bin = motion_bin.type(torch.int)
        
        # Compute toeplitz kernels
        self.toep_kerns = None
        
        # Save
        self.im_size = im_size
        self.use_toeplitz = use_toeplitz
        self.trj = trj
        self.phi = phi
        self.mps = mps
        self.dcf = dcf
        self.tes = tes
        self.b0 = b0
        self.motion_bin = motion_bin
        self.nufft = nufft
        self.imperf_model_motion = imperf_model_motion
        self.bparams = bparams
        self.torch_dev = torch_dev

    def forward(self,
                alphas: torch.Tensor) -> torch.Tensor:
        """
        Forward call of this linear model.

        Parameters
        ----------
        alphas : torch.tensor <complex> | GPU
            the subspace coefficient volumes with shape (nsub, *im_size)
        
        Returns
        ---------
        ksp : torch.tensor <complex> | GPU
            the k-space data with shape (nc, npe, ntr, nro)
        """
       
        # Useful constants
        nt = self.phi.shape[1]
        nc = self.mps.shape[0]
        nmot = self.b0.shape[0]
        motion_bin = self.motion_bin
        coil_batch_size = self.bparams.coil_batch_size
        seg_batch_size = self.bparams.field_batch_size

        # Result array
        ksp = torch.zeros((nc, *self.trj.shape[:-1]), dtype=torch.complex64, device=self.torch_dev)

        # Batch over coils
        for c, d in batch_iterator(nc, coil_batch_size):
            # mps = self.mps[c:d]
            # Batch over time segments
            for l1, l2 in batch_iterator(nt, seg_batch_size):
                
                # phi
                Px = einsum(alphas,self.phi[:,l1:l2], 'nsub nro npe ntr, nsub nt -> nt nro npe ntr')
                # phs 
                phs = torch.exp(1j * 2 * torch.pi * einsum(self.b0, self.tes[l1:l2],'nmot nro npe ntr, nt -> nmot nt nro npe ntr'))
                BPx = einsum(Px,phs, 'nt nro npe ntr, nmot nt nro npe ntr -> nmot nt nro npe ntr')

                # motion correction
                if self.imperf_model_motion is None:
                    TBPx = BPx
                else:
                    TBPx = self.imperf_model_motion.apply_spatial(BPx)

                # sensitivity
                STBPx = einsum(TBPx, self.mps[c:d,...], 'nmot nt nro npe ntr, nc nro npe ntr -> nmot nt nc  nro npe ntr')
                # print(STBPx.shape)
                # print(self.trj[None,motion_bin==0, l1:l2,...].shape)
                for i in range(nmot):
                    # nufft
                    FSTBPx = self.nufft.forward(STBPx[None,i,0,...], self.trj[None,motion_bin==i, l1:l2,...])[0]
                    # Append to k-space
                    ksp[c:d, motion_bin==i,l1:l2,...] = FSTBPx

        return ksp
    
    def adjoint(self,
                ksp: torch.Tensor) -> torch.Tensor:

        # Useful constants
        nt = self.phi.shape[1]
        nsub = self.phi.shape[0]
        nc = self.mps.shape[0]
        nmot = self.b0.shape[0]
        motion_bin = self.motion_bin
        coil_batch_size = self.bparams.coil_batch_size
        seg_batch_size = self.bparams.field_batch_size

        alphas = torch.zeros((nsub, *self.im_size), dtype=torch.complex64, device=self.torch_dev)  
        # Batch over coils
        for c, d in batch_iterator(nc, coil_batch_size):
            mps = self.mps[c:d]
            # Batch over time segments
            for l1, l2 in batch_iterator(nt, seg_batch_size):
                ksp_weighted = ksp[c:d,:,l1:l2,...] 
                len_t = l2-l1
                len_c = d-c
                
                FWy = torch.zeros((nmot, len_t, len_c, *self.mps.shape[1:]), dtype=torch.complex64, device=self.torch_dev)
                # # Batch over motion states
                # for a, b in batch_iterator(nmot, motion_batch_size):
                for i in range(nmot):
                    FWy[i,...] = self.nufft.adjoint(ksp_weighted[None,:,motion_bin==i, ...], self.trj[None, motion_bin==i,l1:l2,...])[0] # nc nsub nseg *im_size
                 # Conjugate maps
                SFWy = einsum(FWy, mps.conj(), 'nmot nt nc ..., nc ... -> nmot nt ...')

                # Conjugate imperfection maps
                if self.imperf_model_motion is None:
                    TSFWy = SFWy
                else:
                    TSFWy = self.imperf_model_motion.apply_spatial_adjoint(SFWy)
                
                phs = torch.exp(1j * 2 * torch.pi * einsum(self.b0, self.tes[l1:l2],'nmot nro npe ntr, nt -> nmot nt nro npe ntr'))
                BTSFWy = einsum(TSFWy,phs.conj(),'nmot nt ..., nmot nt ... -> nt ...')
                PBTSFWy = einsum(BTSFWy,self.phi[:,l1:l2], 'nt nro npe ntr, nsub nt -> nsub nro npe ntr')

                # Append to image
                alphas+= PBTSFWy

        return alphas
    
    def normal(self,
               alphas: torch.Tensor) -> torch.Tensor:
        

        return self.adjoint(self.forward(alphas))
    
    
# NW, 20241220, use cartesian fft for recon
def product(array, dims):
    return reduce(operator.mul, [array[d] for d in dims], 1)

def fftc(x, dim):
    # Same maths as the original one-liner
    #   fftshift(fftn(ifftshift(x))/np.sqrt(product(x.shape, dim)))
    # but the scaling is done IN PLACE, so this allocates 3 intermediates instead
    # of 4, and rebinding x drops each temporary's last reference immediately so
    # the caching allocator can reuse the block. BIT-IDENTICAL to the original
    # (norm='ortho' would also save the allocation but scales inside the
    # transform, which changes the rounding by ~1e-7 relative).
    x_dim = x.shape
    x = torch.fft.ifftshift(x, dim=dim)
    x = torch.fft.fftn(x, dim=dim)
    x /= np.sqrt(product(x_dim, dim))
    return torch.fft.fftshift(x, dim=dim)

def ifftc(x, dim):
    # In-place scaling, as in fftc: 3 intermediates instead of 4, bit-identical.
    x_dim = x.shape
    x = torch.fft.ifftshift(x, dim=dim)
    x = torch.fft.ifftn(x, dim=dim)
    x *= np.sqrt(product(x_dim, dim))
    return torch.fft.fftshift(x, dim=dim)

def _to_xf(x: torch.Tensor) -> torch.Tensor:
    """(..., ny, nz, nx) -> contiguous (..., nx, ny, nz): the x-first layout the
    Cartesian subspace linops compute in."""
    return x.movedim(-1, -3).contiguous()


def _from_xf(x: torch.Tensor) -> torch.Tensor:
    """inverse of _to_xf."""
    return x.movedim(-3, -1).contiguous()


def _xf_zeros(n: int, xf_size: tuple, dev: torch.device) -> list:
    """n separate x-first volumes. An image accumulated as a LIST of
    components can be converted back by _from_xf_list one component at a time,
    so the conversion costs one volume of extra memory instead of a whole
    (nsub, ...) image -- 4.45 GiB at rank 6 on 480x432x480."""
    return [torch.zeros(xf_size, dtype=torch.complex64, device=dev) for _ in range(n)]


def _from_xf_list(parts: list) -> torch.Tensor:
    """stack x-first components into (n, ny, nz, nx), freeing each as it goes."""
    nx, ny, nz = parts[0].shape
    out = torch.empty((len(parts), ny, nz, nx), dtype=parts[0].dtype, device=parts[0].device)
    for s in range(len(parts)):
        out[s] = parts[s].movedim(-3, -1)
        parts[s] = None
    return out


class subspace_cart_linop(linop):

    def __init__(self,
                 im_size: tuple,
                 mask_sample: torch.Tensor,
                 mps: torch.Tensor,
                 phi: torch.Tensor,
                 motion_bin: torch.Tensor,
                 dcf: Optional[torch.Tensor] = None,
                 imperf_model: Optional[tuple] = None,
                 imperf_model_motion: Optional[imperfection] = None,
                 use_toeplitz: Optional[bool] = False,
                 bparams: Optional[batching_params] = batching_params(),
                 kx_fft: Optional[bool] = False,
                 mask_kx: Optional[torch.Tensor] = None,
                 compute_dev: Optional[torch.device] = None):
        """
        Parameters
        ----------
        im_size : tuple 
            image dims as tuple of ints (dim1, dim2, ...)
        mask_sample : torch.tensor <float> | GPU
            The k-space trajectory with shape (npe, ntr, nro, d). 
                we assume that trj values are in [-n/2, n/2] (for nxn grid)
        mps : torch.tensor <complex> | GPU
            sensititvity maps with shape (ncoil, ndim1, ..., ndimN)
        phi : torch.tensor <complex> | GPU
            subspace basis with shape (nsub, ntr)   
        motion_bin : torch.tensor <int> | GPU
            which pe belongs to which motion states (1,npe)
        dcf : torch.tensor <float> | GPU
            the density comp. functon with shape (nro, ...)
        nufft : NUFFT
            the nufft object, defaults to torchkbnufft
        imperf_model : a struct cattying B0 and TEs
            
        imperf_model_motion : motion operator
        use_toeplitz : bool
            toggles toeplitz normal operator
        bparams : batching_params
            contains the batch sizes for the coils, subspace coeffs, and field segments
        kx_fft : bool
            False: the readout axis of the data is x in image space (hybrid
            ky,kz,x data, today's toPython_recon.m output), so the operator
            transforms (ky,kz) only. True: the readout axis is kx, and the
            operator is the full centred 3D FFT -- the x transform is applied to
            the readout of each acquired sample.
        mask_kx : torch.tensor <bool> | GPU, optional
            (nsamp, nx) acquired-kx mask for kx-undersampled data; requires
            kx_fft. None means every kx of every acquired sample was measured,
            which is what lets normal() drop the x transform pair exactly.
        """
        
        ishape = (phi.shape[0], *im_size)
        oshape = (mps.shape[0], motion_bin.shape[0],mps.shape[-1])
        super().__init__(ishape, oshape)

        # Consts
        # compute_dev lets mps live somewhere other than the compute device
        # (host memory), so the full coil array never enters GPU memory --
        # see _mps_batch. Defaults to mps.device, i.e. today's behaviour.
        torch_dev = compute_dev if compute_dev is not None else mps.device
        assert phi.device == torch_dev
        assert mps.device == torch_dev or compute_dev is not None, \
            'mps must be on the compute device unless compute_dev is given'

        # Default params
        # if nufft is None:
        #     nufft = torchkb_nufft(im_size, torch_dev.index)
        if dcf is None:
            dcf = torch.ones((motion_bin.shape[0],mps.shape[-1]), dtype=torch.float32, device=torch_dev)
        else:
            assert dcf.device == torch_dev

        # imperf_model[0] (the "b0" slot) accepts either:
        #   - a raw tensor (nmot, *im_size)               -- exact per-echo exponential (legacy)
        #   - a list/tuple of `imperfection` instances,     -- e.g. main_field_imperfection, one per
        #     one per motion state                            motion state, giving a low-rank
        #                                                      (SVD/time-segmented) B0 phase model
        # instead. imperf_model[1] (tes) is ignored in the latter case since each imperfection
        # instance already encodes its own per-echo time axis internally.
        self.b0_imperf_list = None
        if imperf_model is None:
            b0 = torch.zeros((1, *im_size), dtype=torch.float32, device=torch_dev)
            tes = torch.zeros((phi.shape[1]), dtype=torch.float32, device=torch_dev)
            phs0 = torch.zeros((1, *im_size), dtype=torch.float32, device=torch_dev)
            phst = torch.zeros((phi.shape[1]), dtype=torch.float32, device=torch_dev)
        elif isinstance(imperf_model[0], (list, tuple)) and len(imperf_model[0]) > 0 \
                and isinstance(imperf_model[0][0], imperfection):
            self.b0_imperf_list = list(imperf_model[0])
            b0 = torch.zeros((len(self.b0_imperf_list), *im_size), dtype=torch.float32, device=torch_dev)
            tes = torch.zeros((phi.shape[1]), dtype=torch.float32, device=torch_dev)
            phs0 = imperf_model[2]
            phst = imperf_model[3]
        else:
            b0 = imperf_model[0]
            tes = imperf_model[1]
            phs0 = imperf_model[2]
            phst = imperf_model[3]

        # Rescale and type cast
        mask_sample = mask_sample.type(torch.float32)
        mask_sample = mask_sample>0
        num_sample = motion_bin.shape[0]
        dcf = dcf.type(torch.float32)
        mps = mps.type(torch.complex64)
        # SHIFT-FREE CENTRED FFT. For even ny/nz,
        #   fftshift(fftn(ifftshift(x)))/sqrt(N) == c * chk * fftn(chk*x, 'ortho')
        # with chk[y,z] = (-1)^(y+z) and c = (-1)^(ny/2+nz/2). The image-side chk
        # and the constant fold into mps once here -- and into conj(mps) on the
        # adjoint side -- leaving ONE k-space multiply per transform. So the
        # centred FFT costs 1 allocation instead of the shifting fftc's 3: at
        # 0.5 mm with coil_batch 3 that is ~4.5 GiB less in forward and again in
        # adjoint, and it removes the differently-sized transients that fragment
        # the caching allocator. Same identity already verified against fftc in
        # subspace_cart_sp_linop (test_spatial_subspace.py check 14).
        # chk is built on mps.device because mps may be host-resident here;
        # chk_k lives on the compute device, where the transforms happen.
        _ny, _nz = mps.shape[-3], mps.shape[-2]
        self.shift_free = (_ny % 2 == 0) and (_nz % 2 == 0)
        if self.shift_free:
            _y = torch.arange(_ny, device=mps.device)
            _z = torch.arange(_nz, device=mps.device)
            _chk = ((-1.0) ** ((_y[:, None] + _z[None, :]) % 2)).type(torch.float32)
            _c = (-1.0) ** ((_ny // 2 + _nz // 2) % 2)
            mps = mps * (_c * _chk[:, :, None])
            self.chk_k = _chk.to(torch_dev)
        else:
            self.chk_k = None
            print('subspace_cart_linop: odd ny/nz -> shifting fftc (3 allocs/call)')
        # x-FIRST INTERNAL LAYOUT. Every volume is held as (..., nx, ny, nz) so
        # the (ky,kz) FFT runs over the two innermost, contiguous axes. Over
        # (-3,-2) of an x-innermost array torch first transposes the data:
        # 9.2 of the 12.8 ms a 2-coil 240x216x240 transform took was that copy
        # (GTX 1080 Ti), vs 3.7 ms x-first. Inputs and outputs keep the
        # (..., ny, nz, nx) convention; see _to_xf / _from_xf.
        mps = _to_xf(mps)
        mps = _host_mps(mps, torch_dev, 'subspace_cart_linop')
        phi = phi.type(torch.complex64)
        # The phase maps and their time axes are REAL (B0 in Hz, echo times,
        # the odd/even phase in radians, phst in {0,1}); cfl inputs arrive as
        # complex64 with a zero imaginary part. Keeping them real lets _phase
        # build one float32 angle and a single phasor per echo.
        b0, tes, phs0, phst = (v.real.float() if v.is_complex() else v.float()
                               for v in (b0, tes, phs0, phst))
        b0, phs0 = _to_xf(b0), _to_xf(phs0)
        motion_bin = motion_bin.type(torch.int)
        nt = phi.shape[1]
        nc = mps.shape[0]
        ny, nz, nx = im_size
        nmot = b0.shape[0]

        # Precompute the stacked low-rank spatial/temporal B0 basis once (not per forward/adjoint
        # call) when using the imperfection-instance path.
        if self.b0_imperf_list is not None:
            spatial_stack = torch.stack(
                [imp.apply_spatial(x=None) for imp in self.b0_imperf_list], dim=0)   # (nmot, L, *im_size)
            temporal_stack = torch.stack(
                [imp.apply_temporal_adjoint(y=None) for imp in self.b0_imperf_list], dim=0)  # (nmot, L, nt)
            self.b0_spatial = _to_xf(spatial_stack.type(torch.complex64))
            self.b0_temporal = temporal_stack.type(torch.complex64)
        else:
            self.b0_spatial = None
            self.b0_temporal = None

        # Compute toeplitz kernels
        self.toep_kerns = None

        # Save
        self.im_size = im_size
        self.use_toeplitz = use_toeplitz
        self.mask_sample = mask_sample
        self.num_sample = num_sample
        self.phi = phi
        self.mps = mps
        self.dcf = dcf
        self.tes = tes
        self.b0 = b0
        self.phs0 = phs0
        self.phst = phst
        self.motion_bin = motion_bin
        self.imperf_model_motion = imperf_model_motion
        # self.nm = phs.shape[0]
        # self.nt = phi.shape[1]
        self.bparams = bparams
        self.torch_dev = torch_dev
        self._one = torch.ones((), dtype=torch.float32, device=torch_dev)
        self._mps_conj = None   # materialised lazily by _normal_echo_outer
        assert mask_kx is None or kx_fft, 'mask_kx needs kx_fft=True'
        self.kx_fft = bool(kx_fft)
        self.mask_kx = None if mask_kx is None else mask_kx.to(torch_dev).bool()

        # SAMPLE BOOKKEEPING, precomputed. forward()/adjoint() derived their
        # slice bounds as mask_sample[...].sum().int() -- 0-dim CUDA tensors used
        # as Python slice bounds, which forces a device->host SYNC on each of the
        # ~nt*nc/coil_batch inner iterations -- and rebuilt a boolean scatter
        # mask every time. All of it is static.
        _msk_a = self.mask_sample.reshape(self.mask_sample.shape[0], nt, -1)
        self._samp_cum = []
        for _i in range(_msk_a.shape[0]):
            _cnt = _msk_a[_i].sum(dim=-1).cpu().tolist()
            _acc, _row = 0, [0]
            for _t in range(nt):
                _acc += int(_cnt[_t]); _row.append(_acc)
            self._samp_cum.append(_row)
        self._adj_idx_cache = {}
        # Per-bin sample ranges. toPython_recon.m stores samples bin by bin, so
        # each bin is a CONTIGUOUS run of the sample axis. When that holds, ksp
        # can be indexed with a plain slice instead of a boolean mask, which
        # turns forward()'s read-modify-write of the whole coil batch (gather
        # 2.22 GiB + scatter 2.22 GiB per echo iteration, ~800 GiB per forward
        # at 0.5 mm) into a direct strided write.
        self._bin_slice = []
        _mb_h = motion_bin.cpu()
        for _i in range(_msk_a.shape[0]):
            _w = torch.nonzero(_mb_h == _i).flatten()
            if _w.numel() and int(_w[-1] - _w[0]) == _w.numel() - 1:
                self._bin_slice.append((int(_w[0]), int(_w[-1]) + 1))
            else:
                self._bin_slice.append(None)
        _contig = all(_b is not None for _b in self._bin_slice)
        print(f'  sample indexing: '
              f'{"contiguous bins, direct slice" if _contig else "boolean mask (FALLBACK, slower)"}'
              f'; {nt} echo offsets precomputed on host, no per-iteration sync')

        # DEVICE AUDIT. mps is now deliberately allowed to live on the host, so
        # the old blanket `assert mps.device == torch_dev` no longer catches a
        # stray CPU tensor elsewhere -- it instead surfaces much later, from
        # deep inside forward(), as
        #   "Expected all tensors to be on the same device ... in method bmm"
        # with no indication of which input is at fault. Name it here instead.
        for _n in ('phi', 'mask_sample', 'motion_bin', 'dcf', 'tes', 'b0',
                   'phs0', 'phst', 'chk_k', 'b0_spatial', 'b0_temporal'):
            _v = getattr(self, _n, None)
            if torch.is_tensor(_v) and _v.device != torch_dev:
                print(f'subspace_cart_linop: WARNING self.{_n} is on {_v.device},'
                      f' not the compute device {torch_dev} -- forward() will fail'
                      f' with a device-mismatch error', flush=True)

    def _fftc(self, x: torch.Tensor) -> torch.Tensor:
        """centred 2D FFT over (ny, nz), the two innermost axes of the x-first
        layout: 1 allocation when shift-free."""
        if not self.shift_free:
            return fftc(x, dim=(-2, -1))
        out = torch.fft.fftn(x, dim=(-2, -1), norm='ortho')
        out *= self.chk_k
        return out

    def _ifftc(self, y: torch.Tensor) -> torch.Tensor:
        """adjoint partner of _fftc. The image-side chk and the constant come
        from conj(mps), which already carries them, so only the k-space one is
        applied here. MUTATES y (the checkerboard is applied in place)."""
        if not self.shift_free:
            return ifftc(y, dim=(-2, -1))
        y *= self.chk_k
        return torch.fft.ifftn(y, dim=(-2, -1), norm='ortho')

    def _fft_pair(self, F: torch.Tensor, msk: torch.Tensor) -> torch.Tensor:
        """F_yz^H M F_yz, the (ky,kz) part of A^H A. Shift-free, the k-space
        checkerboard of _fftc and _ifftc meet as chk_k^2 == 1 and drop out, so
        only the mask multiply remains between the two transforms."""
        if not self.shift_free:
            F = fftc(F, dim=(-2, -1))
            F *= msk
            return ifftc(F, dim=(-2, -1))
        F = torch.fft.fftn(F, dim=(-2, -1), norm='ortho')
        F *= msk
        return torch.fft.ifftn(F, dim=(-2, -1), norm='ortho')

    def _sample_idx(self, i: int, l1: int, l2: int):
        """(echo-in-block, (ky,kz) position) of bin i's samples in echoes l1:l2,
        in sample-axis order. Static, so cached: the same (i, l1, l2) recurs for
        every coil batch and every CG iteration."""
        key = (i, l1, l2)
        tp = self._adj_idx_cache.get(key)
        if tp is None:
            idx = torch.nonzero(self.mask_sample[i, l1:l2, ...].flatten()).flatten()
            P_ = self.mask_sample.shape[-2] * self.mask_sample.shape[-1]
            tp = (idx // P_, idx % P_)
            self._adj_idx_cache[key] = tp
        return tp

    def _phase(self, l1: int, l2: int) -> torch.Tensor:
        """exp(i (2 pi b0 te + phs0 phst)) for echoes l1:l2, (nmot, nte, nx, ny, nz).

        The angle is built in float32 and turned into a phasor once, rather
        than in complex arithmetic followed by a complex exp."""
        if self.b0_spatial is not None:
            phs = einsum(self.b0_spatial, self.b0_temporal[:, :, l1:l2],
                         'nmot L ..., nmot L nt -> nmot nt ...')
            ang = einsum(self.phs0, self.phst[l1:l2], 'nmot ..., nt -> nmot nt ...')
            return phs * torch.polar(self._one, ang)
        ang = einsum(self.b0, self.tes[l1:l2] * (2 * torch.pi),
                     'nmot ..., nt -> nmot nt ...')
        ang += einsum(self.phs0, self.phst[l1:l2], 'nmot ..., nt -> nmot nt ...')
        return torch.polar(self._one, ang)

    def _readout_fwd(self, y: torch.Tensor) -> torch.Tensor:
        """hybrid (ky,kz,x) samples -> the measured readout domain.

        With kx_fft the last axis of the data is kx, and this applies the x
        part of the centred 3D FFT -- separable, so F_3D = F_x F_yz -- to the
        readout of each acquired sample (nsamp*nx points, not the full nt*ny*nz
        grid). mask_kx then zeroes the kx points that were not acquired."""
        if self.kx_fft:
            y = fftc(y, dim=(-1,))
        if self.mask_kx is not None:
            y = y * self.mask_kx
        return y

    def _readout_adj(self, y: torch.Tensor) -> torch.Tensor:
        """adjoint of _readout_fwd."""
        if self.mask_kx is not None:
            y = y * self.mask_kx
        if self.kx_fft:
            y = ifftc(y, dim=(-1,))
        return y

    def forward(self,
                alphas: torch.Tensor) -> torch.Tensor:

        nt = self.phi.shape[1]
        nc = self.mps.shape[0]
        ny, nz, nx = self.im_size
        nmot = self.b0.shape[0]
        motion_bin = self.motion_bin
        coil_batch_size = self.bparams.coil_batch_size
        seg_batch_size = self.bparams.field_batch_size
        alphas = _to_xf(alphas)

        ksp = torch.zeros((nc, self.num_sample, nx), dtype=torch.complex64, device=self.torch_dev)

        for c, d in batch_iterator(nc, coil_batch_size):
            # hoisted OUT of the echo loop below: with a host-resident mps this
            # is a real 1.59 GB copy per coil batch
            mps_c = _mps_batch(self, c, d)
            for l1, l2 in batch_iterator(nt, seg_batch_size):
                len_t = l2 - l1
                Px = einsum(alphas, self.phi[:, l1:l2], 'nsub ..., nsub nt -> nt ...')
                BPx = self._phase(l1, l2) * Px[None]                 # nmot nt nx ny nz
                del Px
                if self.imperf_model_motion is not None:
                    BPx = _to_xf(self.imperf_model_motion.apply_spatial(_from_xf(BPx)))
                F = self._fftc(BPx[:, None] * mps_c[None, :, None])   # nmot nc nt nx ny nz
                del BPx

                for i in range(nmot):
                    # only the sampled (echo, ky, kz) are read, straight into
                    # their slots in ksp; st/ed are host ints (no device sync)
                    G = F[i].reshape(d - c, len_t, nx, ny * nz)
                    tl, pp = self._sample_idx(i, l1, l2)
                    vals = G[:, tl, :, pp].transpose(0, 1)                # nc n nx
                    st = self._samp_cum[i][l1]
                    ed = self._samp_cum[i][l2]
                    sl = self._bin_slice[i]
                    if sl is None:
                        tmp = ksp[c:d, motion_bin == i, ...]
                        tmp[:, st:ed, ...] = vals
                        ksp[c:d, motion_bin == i, :] = tmp
                    else:
                        ksp[c:d, sl[0] + st:sl[0] + ed, :] = vals
                del F
            if self.kx_fft:
                ksp[c:d] = self._readout_fwd(ksp[c:d])

        return ksp

    def adjoint(self,
                ksp: torch.Tensor) -> torch.Tensor:

        nt = self.phi.shape[1]
        nsub = self.phi.shape[0]
        nc = self.mps.shape[0]
        ny, nz, nx = self.im_size
        nmot = self.b0.shape[0]
        motion_bin = self.motion_bin
        coil_batch_size = self.bparams.coil_batch_size
        seg_batch_size = self.bparams.field_batch_size
        alphas = _xf_zeros(nsub, (nx, ny, nz), self.torch_dev)
        for c, d in batch_iterator(nc, coil_batch_size):
            # adjoint only ever uses conj(mps), so materialise that ONCE per
            # coil batch and never hold the unconjugated copy
            mps_conj = _mps_batch(self, c, d).conj().resolve_conj()
            # a host-resident ksp is copied in once per coil batch
            ksp_c = self._readout_adj(_ksp_batch(self, ksp, c, d))

            for l1, l2 in batch_iterator(nt, seg_batch_size):
                len_t = l2 - l1
                SFWy = torch.zeros((nmot, len_t, nx, ny, nz), dtype=torch.complex64, device=self.torch_dev)
                for i in range(nmot):
                    st = self._samp_cum[i][l1]
                    ed = self._samp_cum[i][l2]
                    sl = self._bin_slice[i]
                    if sl is None:
                        src = ksp_c[:, motion_bin == i, ...][:, st:ed, ...]
                    else:
                        src = ksp_c[:, sl[0] + st:sl[0] + ed, :]
                    kw = torch.zeros((d - c, len_t, nx, ny * nz), dtype=torch.complex64, device=self.torch_dev)
                    tl, pp = self._sample_idx(i, l1, l2)
                    kw[:, tl, :, pp] = src.transpose(0, 1)
                    FWy = self._ifftc(kw.reshape(d - c, len_t, nx, ny, nz))
                    del kw
                    FWy *= mps_conj[:, None]
                    SFWy[i, ...] = FWy.sum(0)
                    del FWy
                if self.imperf_model_motion is not None:
                    SFWy = _to_xf(self.imperf_model_motion.apply_spatial_adjoint(_from_xf(SFWy)))
                SFWy *= self._phase(l1, l2).conj()
                BTSFWy = SFWy.sum(0)
                del SFWy
                # NOTE: must contract with conj(phi), not phi -- forward()
                # expands the image with plain phi, so the adjoint of "sum over
                # nsub weighted by phi" is "sum over nt weighted by conj(phi)".
                ph = self.phi[:, l1:l2].conj()
                for s in range(nsub):
                    alphas[s] += einsum(BTSFWy, ph[s], 'nt ..., nt -> ...')
                del BTSFWy

        return _from_xf_list(alphas)

    def normal(self,
               alphas: torch.Tensor) -> torch.Tensor:
        """A^H A without ever forming the sample-space vector.

        A^H A = sum over (coil, echo, bin) of the corresponding piece: the
        forward is independent per coil, and the sampling mask partitions
        samples by (echo, bin), so those sums are independent too. Ma^H Ma is
        then simply the 0/1 mask -- a MULTIPLY, with no gather and no scatter.

        With kx_fft and no mask_kx the readout transform cancels exactly
        (F_x^H F_x = I, and the (ky,kz,t) mask does not depend on kx), so this
        is the same operator for hybrid and kx-space data. A kx mask, or a
        motion operator (which does not commute through the echo split), falls
        back to the coil-batched adjoint(forward()).
        """
        if self.imperf_model_motion is not None or self.mask_kx is not None:
            return self._normal_via_fwd_adj(alphas)
        if self.mps.device != self.torch_dev:
            return _from_xf_list(self._normal_coil_outer(alphas))
        # x-first copy of the input: +1 image for the duration of the call.
        # Only on this path, where the coil maps fit on the GPU anyway; the
        # output costs one extra volume either way (see _from_xf_list)
        a = _to_xf(alphas)
        out = self._normal_echo_outer(a)
        del a
        return _from_xf_list(out)

    def _normal_echo_outer(self,
                           alphas: torch.Tensor) -> torch.Tensor:
        """Echo loop outside, coil loop inside: the per-echo work (B0 phase,
        subspace expansion, phase conjugate, subspace projection) is done once
        per echo instead of once per (echo, coil batch). Needs mps on the
        compute device -- a hosted mps would cross PCIe once per echo.
        x-first in, list of x-first components out."""
        nc = self.mps.shape[0]
        nt, nsub = self.phi.shape[1], self.phi.shape[0]
        cb = self.bparams.coil_batch_size
        fb = self.bparams.field_batch_size
        if self._mps_conj is None:
            self._mps_conj = self.mps.conj().resolve_conj()

        out = _xf_zeros(nsub, alphas.shape[1:], self.torch_dev)
        for l1, l2 in batch_iterator(nt, fb):
            phs = self._phase(l1, l2)                                  # nmot nt nx ny nz
            Px = einsum(alphas, self.phi[:, l1:l2], 'nsub ..., nsub nt -> nt ...')
            BPx = phs * Px[None]
            del Px
            msk = self.mask_sample[:, None, l1:l2, None, :, :]        # nmot 1 nt 1 ny nz
            acc = torch.zeros_like(BPx)
            for c, d in batch_iterator(nc, cb):
                F = self._fft_pair(BPx[:, None] * self.mps[None, c:d, None], msk)
                F *= self._mps_conj[None, c:d, None]
                acc += F.sum(1)
                del F
            acc *= phs.conj()
            BT = acc.sum(0)
            del acc, phs, BPx
            ph = self.phi[:, l1:l2].conj()
            for s in range(nsub):
                out[s] += einsum(BT, ph[s], 'nt ..., nt -> ...')
            del BT
        return out

    def _normal_coil_outer(self,
                           alphas: torch.Tensor) -> torch.Tensor:
        """Coil batch outermost, so a host-resident mps crosses PCIe once per
        coil batch rather than once per echo. Per-echo work is repeated per
        coil batch -- the price of streaming the maps.

        Takes the caller's x-LAST alphas and transposes only the per-echo
        expansion Phi^T x (one volume per echo batch), so this path -- the one
        taken when memory is tight -- holds no x-first copy of the image. Out:
        list of x-first components."""
        nc = self.mps.shape[0]
        nt, nsub = self.phi.shape[1], self.phi.shape[0]
        cb = self.bparams.coil_batch_size
        fb = self.bparams.field_batch_size
        ny, nz, nx = self.im_size

        out = _xf_zeros(nsub, (nx, ny, nz), self.torch_dev)
        for c, d in batch_iterator(nc, cb):
            mps_c = _mps_batch(self, c, d)
            mps_c_conj = mps_c.conj().resolve_conj()
            for l1, l2 in batch_iterator(nt, fb):
                phs = self._phase(l1, l2)
                Px = _to_xf(einsum(alphas, self.phi[:, l1:l2], 'nsub ..., nsub nt -> nt ...'))
                F = (phs * Px[None])[:, None] * mps_c[None, :, None]   # nmot nc nt nx ny nz
                del Px
                F = self._fft_pair(F, self.mask_sample[:, None, l1:l2, None, :, :])
                F *= mps_c_conj[None, :, None]
                BT = F.sum(1)
                del F
                BT *= phs.conj()
                BT = BT.sum(0)
                del phs
                # accumulate ONE subspace component at a time: the
                # single-einsum form builds a full (nsub, *im_size) temporary
                ph = self.phi[:, l1:l2].conj()
                for s in range(nsub):
                    out[s] += einsum(BT, ph[s], 'nt ..., nt -> ...')
                del BT
        return out

    def _normal_via_fwd_adj(self,
                            alphas: torch.Tensor) -> torch.Tensor:
        """Coil-batched adjoint(forward()). Used when a motion operator makes the
        fused form above invalid; keeps the sample-space tensor at one coil
        batch rather than all coils."""
        nc = self.mps.shape[0]
        cb = self.bparams.coil_batch_size
        if cb >= nc:
            return self.adjoint(self.forward(alphas))
        out = None
        mps_all, oshape_all = self.mps, self.oshape
        try:
            for c, d in batch_iterator(nc, cb):
                self.mps = mps_all[c:d]
                self.oshape = (d - c, *oshape_all[1:])
                part = self.adjoint(self.forward(alphas))
                out = part if out is None else out + part
        finally:
            self.mps, self.oshape = mps_all, oshape_all
        return out

def _dev_batch(t: torch.Tensor, c: int, d: int, dev: torch.device) -> torch.Tensor:
    """t[c:d] on `dev`, copying only if it is not already there."""
    b = t[c:d]
    return b if b.device == dev else b.to(dev, non_blocking=True)


def _ksp_batch(lin, ksp: torch.Tensor, c: int, d: int) -> torch.Tensor:
    """k-space for coil batch [c, d) on the compute device.

    Lets the MEASURED data stay in host memory: adjoint() has the coil loop
    outermost, so only one batch is resident -- (nc_b, nsamp, nx) rather than
    (nc, nsamp, nx), i.e. 1.59 GiB instead of 12.74 GiB at 0.5 mm / 16ch. Only
    adjoint() reads the measured data (CG and FISTA both use it solely for
    A^H b), so nothing else has to change.
    """
    return _dev_batch(ksp, c, d, lin.torch_dev)


def _mps_batch(lin, c: int, d: int) -> torch.Tensor:
    """Coil maps for batch [c, d) on the linop's COMPUTE device.

    When a linop is built with `compute_dev` set and `mps` left in host memory,
    the full (nc, ny, nz, nx) coil array never occupies GPU memory -- only the
    batch currently being worked on is copied in. At 0.5 mm with 16 channels
    that keeps 12.7 GB off the GPU in exchange for nc_batch * 0.8 GB resident
    and one transfer per coil batch per forward/adjoint call.

    Unlike x-slabbing this does not touch the operator's structure at all, so it
    stays valid for a whole-volume 3D recon with x-wise undersampling.

    A no-op when mps already lives on the compute device, so every existing
    caller is unaffected.
    """
    return _dev_batch(lin.mps, c, d, lin.torch_dev)


def _host_mps(mps: torch.Tensor, torch_dev: torch.device, tag: str) -> torch.Tensor:
    """Prepare a host-resident mps: contiguous (so each batch copy is a single
    contiguous DMA) and pinned where possible (so the copy can be async).

    Pinning is best-effort: page-locking 12.7 GB can fail, and a failure only
    costs transfer speed, never correctness.
    """
    if mps.device == torch_dev:
        return mps
    mps = mps.contiguous()
    gb = mps.numel() * mps.element_size() / 1e9
    pinned = False
    if mps.device.type == 'cpu' and torch.cuda.is_available():
        try:
            mps = mps.pin_memory()
            pinned = True
        except RuntimeError as e:
            print(f'  {tag}: could not pin {gb:.1f} GB of coil maps ({e}); '
                  f'transfers will be synchronous')
    print(f'  {tag}: mps stays on {mps.device} ({gb:.1f} GB'
          f'{", pinned" if pinned else ""}), streamed per coil batch to '
          f'{torch_dev}')
    return mps


# -----------------------------------------------------------------------------
# TEMPORAL GRAM (deduplicated). The sampling mask folds into the temporal basis:
# for a Cartesian acquisition a sample is one (echo, ky, kz), so at a fixed
# (ky,kz) only the echo set T_p actually acquired there contributes, and
#     Gram[p][a', a] = sum_{t in T_p} cnt_t * conj(u[a', t]) * u[a, t]
# is one A x A matrix per position, diagonal in (ky,kz). Gram[p] depends only on
# T_p, and an EPTI trajectory has very few distinct echo sets, so it is stored
# deduplicated: unique matrices plus one index per position.
def _dedup_temporal_gram(basis: torch.Tensor,
                         cnt: np.ndarray,
                         torch_dev: torch.device):
    """Unique temporal Gram matrices and a per-position index.

    basis : (A, nt) complex   combined temporal vectors
    cnt   : (P, nt) integer   how many times echo t is acquired at position p
                              (a COUNT, not a boolean: A^H A sums over bins, so
                              an echo two bins both acquire contributes twice)

    Returns (gu, inv) with gu (n_uniq, A, A) complex and inv (P,) int64 such
    that the dense Gram would be gu[inv]. Positions with an empty echo set get
    an all-zero matrix, which is correct -- they contribute nothing.
    """
    A, nt = basis.shape
    uniq, inv = np.unique(cnt, axis=0, return_inverse=True)
    gu = torch.zeros((uniq.shape[0], A, A), dtype=torch.complex64, device=torch_dev)
    for u in range(uniq.shape[0]):
        tsel = np.nonzero(uniq[u])[0]
        if tsel.size == 0:
            continue
        Bu = basis[:, torch.from_numpy(tsel).to(torch_dev)]
        w = torch.from_numpy(uniq[u][tsel].astype(np.float32)).to(torch_dev)
        # Gram[a',a] = sum_t cnt_t conj(u[a',t]) u[a,t] = conj(Bu) diag(w) Bu^T
        # NOT Bu @ Bu^H, which is its conjugate (~0.8 rel err on a complex basis)
        gu[u] = (Bu.conj() * w) @ Bu.T
    return gu, torch.from_numpy(inv.reshape(-1)).to(torch.int64).to(torch_dev)


def _dedup_gram_order(inv: torch.Tensor, n_uniq: int):
    """Sort positions by which unique Gram they use, so each group is a
    contiguous slice and the apply is one batched matmul per unique matrix
    (~50 of them) instead of a per-position gather of a (P, A, A) tensor.

    Returns (perm, groups) with perm (P,) int64 and groups a host list of
    (start, stop, u) covering perm.
    """
    inv_h = inv.cpu().numpy()
    perm_h = np.argsort(inv_h, kind='stable')
    sorted_u = inv_h[perm_h]
    bounds = np.searchsorted(sorted_u, np.arange(n_uniq + 1))
    groups = [(int(bounds[u]), int(bounds[u + 1]), u)
              for u in range(n_uniq) if bounds[u + 1] > bounds[u]]
    return torch.from_numpy(perm_h).to(torch.int64).to(inv.device), groups


def _apply_dedup_gram(gu: torch.Tensor, perm: torch.Tensor, groups: list,
                      X: torch.Tensor) -> torch.Tensor:
    """Apply Gram[p] over the A axis of X (nc, A, P, nx), IN PLACE.

    One batched matmul per unique Gram, touching only that group's positions.
    In-place is safe because the Gram is diagonal in p and the groups partition
    p, so a group's output depends only on its own input positions.
    """
    for s, e, u in groups:
        idx = perm[s:e]
        X[:, :, idx, :] = einsum(gu[u], X[:, :, idx, :],
                                 'a b, nc b n x -> nc a n x')
    return X


class subspace_cart_linop_fastb0(linop):
    """
    Same forward model as subspace_cart_linop, restructured so the (ny,nz) FFT batch scales with
    (nsub * B0_L * phs2_L) instead of nt: the B0/phs2 low-rank SPATIAL bases are multiplied into
    the coil-weighted image and FFT'd once, and the nt-dependence (via phi / B0_temporal /
    phs2_temporal) is contracted back in AFTER the FFT via a cheap per-sample weight -- exactly
    the ordering subspace_linop (the NUFFT-based linop in linops.py) already uses for the NUFFT
    case. Real speedup requires nsub*B0_L*phs2_L << nt.

    Requires the B0 phase (and, if present, the phs_diff0/phst correction) as `imperfection`
    instances with the low-rank machinery already built (e.g. combined_exact_tes from
    main_field_nw.py). Pass phs2_imperf_list=None if there is no separate correction.

    Does NOT support imperf_model_motion: a rigid-body/warp motion correction operates on the
    full per-echo image and doesn't have the product-separable (spatial(r) * temporal(t))
    structure a multiplicative phase does, so it can't be reordered before the FFT.

    kx_fft / mask_kx: as in subspace_cart_linop -- with kx_fft the readout axis of the data is
    kx and the operator is the full centred 3D FFT; the x transform is applied to the readout of
    each acquired sample, and normal() drops it exactly (F_x^H F_x = I) unless a kx mask is given.

    Memory: bparams.field_batch_size controls how many B0_L components go through the FFT at
    once. Each batch materializes a (coil_batch, nsub, field_batch_size*phs2_L, *im_size) tensor
    before the FFT -- ~300 MB per (coil, B0 component) at nsub=3 on 240x216x240.
    """

    def __init__(self,
                 im_size: tuple,
                 mask_sample: torch.Tensor,
                 mps: torch.Tensor,
                 phi: torch.Tensor,
                 motion_bin: torch.Tensor,
                 b0_imperf_list: list,
                 phs2_imperf_list: Optional[list] = None,
                 bparams: Optional[batching_params] = batching_params(),
                 gram_normal: Optional[str] = 'off',
                 gram_max_gb: Optional[float] = 2.0,
                 kx_fft: Optional[bool] = False,
                 mask_kx: Optional[torch.Tensor] = None,
                 compute_dev: Optional[torch.device] = None):
        ishape = (phi.shape[0], *im_size)
        oshape = (mps.shape[0], motion_bin.shape[0], mps.shape[-1])
        super().__init__(ishape, oshape)

        # compute_dev lets mps live somewhere other than the compute device
        # (host memory), so the full coil array never enters GPU memory --
        # see _mps_batch. Defaults to mps.device, i.e. today's behaviour.
        torch_dev = compute_dev if compute_dev is not None else mps.device
        nmot = len(b0_imperf_list)
        nt = phi.shape[1]
        ny, nz = im_size[0], im_size[1]

        mask_sample = (mask_sample.type(torch.float32) > 0).to(torch_dev)
        mps = mps.type(torch.complex64)
        # SHIFT-FREE CENTRED FFT, as in subspace_cart_linop: for even ny/nz
        #   fftc(x) == c * chk * fftn(chk * x, 'ortho'),  chk[y,z] = (-1)^(y+z)
        # The image-side chk and c fold into mps here; the k-space chk is a +-1
        # per SAMPLE, so it folds into the per-sample temporal weight below and
        # costs nothing per call. In the temporal Gram it cancels (chk^2 == 1).
        self.shift_free = (ny % 2 == 0) and (nz % 2 == 0)
        if self.shift_free:
            _y = torch.arange(ny, device=mps.device)
            _z = torch.arange(nz, device=mps.device)
            _chk = ((-1.0) ** ((_y[:, None] + _z[None, :]) % 2)).type(torch.float32)
            _c = (-1.0) ** ((ny // 2 + nz // 2) % 2)
            mps = mps * (_c * _chk[:, :, None])
            chk_k = _chk.to(torch_dev).flatten()
        # x-first internal layout, as in subspace_cart_linop: the (ky,kz) FFT
        # then runs over the innermost axes without a transpose copy
        mps = _to_xf(mps)
        mps = _host_mps(mps, torch_dev, 'subspace_cart_linop_fastb0')
        phi = phi.type(torch.complex64).to(torch_dev)
        motion_bin = motion_bin.type(torch.int).to(torch_dev)

        # (nmot, B0_L, nx, ny, nz), filled bin by bin straight into the x-first
        # layout: stacking first and permuting after would hold the 4.5 GB
        # (1 mm, B0_L=45) basis twice on top of the imperfection objects' copy
        L0 = b0_imperf_list[0].apply_spatial(x=None).shape[0]
        b0_spatial = torch.empty((nmot, L0, im_size[2], ny, nz), dtype=torch.complex64, device=torch_dev)
        for i_, imp in enumerate(b0_imperf_list):
            b0_spatial[i_] = imp.apply_spatial(x=None).movedim(-1, -3)
        b0_temporal = torch.stack([imp.apply_temporal_adjoint(y=None) for imp in b0_imperf_list], dim=0)  # (nmot, B0_L, nt)
        b0_temporal = b0_temporal.reshape(nmot, b0_temporal.shape[1], -1)

        # phs2 is optional; None means "no separate correction", i.e. a rank-1
        # all-ones term. It is kept as None rather than materialised so neither
        # the FFT input build nor the adjoint multiplies by a volume of ones.
        if phs2_imperf_list is None:
            phs2_spatial = phs2_temporal = None
        else:
            phs2_spatial = torch.stack([imp.apply_spatial(x=None) for imp in phs2_imperf_list], dim=0)
            phs2_temporal = torch.stack([imp.apply_temporal_adjoint(y=None) for imp in phs2_imperf_list], dim=0)
            phs2_spatial = _to_xf(phs2_spatial.type(torch.complex64).to(torch_dev))
            phs2_temporal = phs2_temporal.reshape(nmot, phs2_temporal.shape[1], -1).type(torch.complex64).to(torch_dev)

        self.im_size = im_size
        self.mask_sample = mask_sample
        self.num_sample = motion_bin.shape[0]
        self.phi = phi
        self.mps = mps
        self.motion_bin = motion_bin
        self.nmot = nmot
        self.ny, self.nz = ny, nz
        self.b0_spatial = b0_spatial
        self._xf_size = (im_size[2], ny, nz)
        self.b0_temporal = b0_temporal.type(torch.complex64).to(torch_dev)
        self.phs2_spatial = phs2_spatial
        self.phs2_temporal = phs2_temporal
        self.B0_L = self.b0_spatial.shape[1]
        self.phs2_L = 1 if phs2_spatial is None else phs2_spatial.shape[1]
        self.bparams = bparams
        self.torch_dev = torch_dev
        assert mask_kx is None or kx_fft, 'mask_kx needs kx_fft=True'
        self.kx_fft = bool(kx_fft)
        self.mask_kx = None if mask_kx is None else mask_kx.to(torch_dev).bool()

        # PER-BIN SAMPLE TABLES, built once. forward()/adjoint() used to redo
        # torch.nonzero(mask) (a device sync) and rebuild the combined temporal
        # weight on every call. Per bin: the flattened (ky,kz) position of each
        # sample, where the bin's samples sit in the sample axis (a slice when
        # contiguous, as toPython_recon.m writes them), and
        #   W[l, b, q, n] = phi[l, t_n] * B0_temporal[b, t_n] * phs2_temporal[q, t_n] * chk(p_n)
        # i.e. everything about a sample that does not depend on space.
        # (nsub, B0_L, phs2_L, nsamp_bin) complex64: ~90 MB per bin at 1 mm.
        self._bins = []
        mb_h = motion_bin.cpu()
        for i in range(nmot):
            t_idx, y_idx, z_idx = torch.nonzero(self.mask_sample[i], as_tuple=True)
            w = torch.nonzero(mb_h == i).flatten()
            if w.numel() and int(w[-1] - w[0]) == w.numel() - 1:
                where = slice(int(w[0]), int(w[-1]) + 1)
            else:
                where = (motion_bin == i)
            if t_idx.numel() == 0:
                self._bins.append(None)
                continue
            flat = y_idx * nz + z_idx
            W = (self.phi[:, t_idx][:, None, None, :]
                 * self.b0_temporal[i][:, t_idx][None, :, None, :])
            if self.phs2_temporal is not None:
                W = W * self.phs2_temporal[i][:, t_idx][None, None, :, :]
            if self.shift_free:
                W = W * chk_k[flat]
            self._bins.append((flat, where, W.contiguous()))

        # ── optional temporal Gram for normal() ────────────────────────────
        # A^H A without ever forming the (nc, nsub, B0_L, phs2_L, nsamp, nx)
        # expanded product: for each (ky,kz) only the echoes acquired there
        # matter, giving one A x A matrix per position with A = nsub*B0_L*phs2_L.
        # The FFT count is UNCHANGED (2*A per coil either way), so the win is
        # confined to the contraction and the gather/scatter -- and its working
        # set is A volumes per coil, which is why it defaults to off.
        # Per-bin, because b0_spatial/phs2_spatial depend on the bin.
        self.gram = None
        assert gram_normal in ('auto', 'on', 'off'), \
            f'unknown gram_normal {gram_normal!r}'
        if gram_normal != 'off':
            A = self.phi.shape[0] * self.B0_L * self.phs2_L
            P_ = self.ny * self.nz
            gu_l, perm_l, grp_l = [], [], []
            n_uniq_tot = 0
            for i in range(nmot):
                # combined temporal vector u[(l,b,q), t], l slowest -- must match
                # the (nsub, B0_L, phs2_L) flatten order normal() uses
                uu = self.phi[:, None, None, :] * self.b0_temporal[i][None, :, None, :]
                if self.phs2_temporal is not None:
                    uu = uu * self.phs2_temporal[i][None, None, :, :]
                uu = uu.reshape(A, nt)
                # echoes acquired at each position, this bin. mask_sample[i] is
                # (nt, ny, nz) boolean, so within one bin each (t,p) occurs once
                occ = self.mask_sample[i].reshape(nt, P_).T.cpu().numpy()
                gu, inv = _dedup_temporal_gram(uu, occ.astype(np.uint16), torch_dev)
                perm, groups = _dedup_gram_order(inv, gu.shape[0])
                gu_l.append(gu); perm_l.append(perm); grp_l.append(groups)
                n_uniq_tot += gu.shape[0]
            gb = sum(g.numel() for g in gu_l) * 8 / 1e9
            dense_gb = nmot * P_ * A * A * 8 / 1e9
            if gb > gram_max_gb:
                print(f'subspace_cart_linop_fastb0: temporal Gram needs {gb:.2f} GB '
                      f'(> gram_max_gb={gram_max_gb}) -> not building it')
            else:
                self.gram = (gu_l, perm_l, grp_l)
                self.gram_A = A
                print(f'  temporal Gram: A = {A} (nsub {self.phi.shape[0]} x B0_L '
                      f'{self.B0_L} x phs2_L {self.phs2_L}), {n_uniq_tot} distinct '
                      f'echo sets over {nmot} bin(s) -> {gb * 1e3:.1f} MB '
                      f'deduplicated vs {dense_gb:.2f} GB dense')
                ncb = min(bparams.coil_batch_size, mps.shape[0])
                work = ncb * A * P_ * im_size[-1] * 8 / 1e9
                print(f'  normal() working set: ({ncb}, {A}, {P_}, {im_size[-1]})'
                      f' = {work:.1f} GB'
                      + ('' if work < 8 else
                         f'  <-- lower coil_batch_size (1 -> {work / ncb:.1f} GB)'
                         f' or B0_L, or use gram_normal=off'))

    def _fft2(self, x: torch.Tensor) -> torch.Tensor:
        if self.shift_free:
            return torch.fft.fftn(x, dim=(-2, -1), norm='ortho')
        return fftc(x, dim=(-2, -1))

    def _ifft2(self, y: torch.Tensor) -> torch.Tensor:
        if self.shift_free:
            return torch.fft.ifftn(y, dim=(-2, -1), norm='ortho')
        return ifftc(y, dim=(-2, -1))

    def _spatial(self, i: int, l1: int, l2: int) -> torch.Tensor:
        """(K, nx, ny, nz) combined spatial maps for B0 components l1:l2 of
        bin i, K = (l2 - l1) * phs2_L in (b, q) order, q fastest."""
        U = self.b0_spatial[i, l1:l2]
        if self.phs2_spatial is None:
            return U
        return (U[:, None] * self.phs2_spatial[i][None]).reshape(-1, *self._xf_size)

    _readout_fwd = subspace_cart_linop._readout_fwd
    _readout_adj = subspace_cart_linop._readout_adj

    def _forward_coils(self, alphas: torch.Tensor, c: int, d: int) -> torch.Tensor:
        """Hybrid-space (ky,kz,x) samples of coils c:d, (d-c, nsamp, nx).
        alphas is x-first."""
        nx = self.im_size[-1]
        nsub = self.phi.shape[0]
        q = self.phs2_L
        mps_cd = _mps_batch(self, c, d)
        ksp = torch.zeros((d - c, self.num_sample, nx), dtype=torch.complex64, device=self.torch_dev)
        for i in range(self.nmot):
            if self._bins[i] is None:
                continue
            flat, where, W = self._bins[i]
            ksp_i = torch.zeros((d - c, nx, flat.numel()), dtype=torch.complex64, device=self.torch_dev)
            for l1, l2 in batch_iterator(self.B0_L, self.bparams.field_batch_size):
                SU = mps_cd[:, None] * self._spatial(i, l1, l2)[None]            # (nc, K, ...)
                F = self._fft2(SU[:, None] * alphas[None, :, None])              # (nc, nsub, K, ...)
                del SU
                F = F.reshape(d - c, nsub * (l2 - l1) * q, nx, -1)
                Wb = W[:, l1:l2].reshape(nsub * (l2 - l1) * q, -1)
                for a in range(Wb.shape[0]):
                    ksp_i += F[:, a].index_select(-1, flat) * Wb[a]
                del F
            ksp[:, where] = ksp_i.transpose(1, 2)
        return ksp

    def _adjoint_coils(self, ksp_c: torch.Tensor, c: int, d: int,
                       out: torch.Tensor) -> torch.Tensor:
        """out += A_c^H ksp_c for coils c:d; ksp_c is hybrid and on device,
        out a list of x-first components (see _xf_zeros)."""
        nx = self.im_size[-1]
        nsub = self.phi.shape[0]
        q = self.phs2_L
        P_ = self.ny * self.nz
        mps_conj = _mps_batch(self, c, d).conj().resolve_conj()
        for i in range(self.nmot):
            if self._bins[i] is None:
                continue
            flat, where, W = self._bins[i]
            y = ksp_c[:, where].transpose(1, 2).contiguous()                     # (nc, nx, n)
            for l1, l2 in batch_iterator(self.B0_L, self.bparams.field_batch_size):
                K = (l2 - l1) * q
                Wb = W[:, l1:l2].reshape(nsub * K, -1).conj()
                Y = torch.zeros((d - c, nsub * K, nx, P_), dtype=torch.complex64, device=self.torch_dev)
                for a in range(nsub * K):
                    Y[:, a].index_add_(-1, flat, y * Wb[a])
                Z = self._ifft2(Y.reshape(d - c, nsub, K, *self._xf_size))
                del Y
                V = mps_conj[:, None] * self._spatial(i, l1, l2).conj()[None]   # (nc, K, ...)
                Z *= V[:, None]
                del V
                Z = Z.sum(dim=(0, 2))
                for s in range(nsub):
                    out[s] += Z[s]
                del Z
        return out

    def forward(self,
                alphas: torch.Tensor) -> torch.Tensor:
        nc = self.mps.shape[0]
        ksp = torch.empty((nc, self.num_sample, self.im_size[-1]),
                          dtype=torch.complex64, device=self.torch_dev)
        alphas = _to_xf(alphas)
        for c, d in batch_iterator(nc, self.bparams.coil_batch_size):
            ksp[c:d] = self._readout_fwd(self._forward_coils(alphas, c, d))
        return ksp

    def adjoint(self,
                ksp: torch.Tensor) -> torch.Tensor:
        nc = self.mps.shape[0]
        out = _xf_zeros(self.phi.shape[0], self._xf_size, self.torch_dev)
        for c, d in batch_iterator(nc, self.bparams.coil_batch_size):
            # ksp may be host-resident (--ksp-on-host): bring this coil batch
            # to the compute device first
            ksp_c = self._readout_adj(_ksp_batch(self, ksp, c, d))
            self._adjoint_coils(ksp_c, c, d, out)
        return _from_xf_list(out)

    def normal(self,
               alphas: torch.Tensor) -> torch.Tensor:
        """A^H A, one coil batch at a time: forward into that batch's samples
        and straight back, so the full (nc, nsamp, nx) k-space is never held.
        The readout transform cancels unless a kx mask is present.

        With the temporal Gram (gram_normal != 'off') the echo axis is never
        expanded: the sampling mask is folded into the temporal basis, so each
        (ky,kz) needs only one A x A matrix, A = nsub*B0_L*phs2_L."""
        alphas = _to_xf(alphas)
        if self.gram is not None:
            return _from_xf(self._normal_gram(alphas))
        nc = self.mps.shape[0]
        out = _xf_zeros(self.phi.shape[0], self._xf_size, self.torch_dev)
        for c, d in batch_iterator(nc, self.bparams.coil_batch_size):
            y = self._forward_coils(alphas, c, d)
            if self.mask_kx is not None:
                y = self._readout_adj(self._readout_fwd(y))
            self._adjoint_coils(y, c, d, out)
            del y
        del alphas
        return _from_xf_list(out)

    def _normal_gram(self,
                     alphas: torch.Tensor) -> torch.Tensor:
        assert self.mask_kx is None, 'temporal Gram does not support mask_kx'
        gu_l, perm_l, grp_l = self.gram
        nc = self.mps.shape[0]
        nx = self.im_size[-1]
        nsub = self.phi.shape[0]
        ny, nz = self.ny, self.nz
        P_, A, K = ny * nz, self.gram_A, self.B0_L * self.phs2_L
        cb = self.bparams.coil_batch_size
        kb = max(1, self.bparams.field_batch_size * self.phs2_L)

        out = torch.zeros_like(alphas)            # x-first in, x-first out
        for c, d in batch_iterator(nc, cb):
            mps_cd = _mps_batch(self, c, d)
            ncb = d - c
            for i in range(self.nmot):
                U = self._spatial(i, 0, self.B0_L)
                # THE memory hog, and it is intrinsic to the Gram: its
                # cross-terms couple all A components at a given (ky,kz), so all
                # A of them must be resident -- A times the ksp the
                # sample-space path holds.
                F = torch.empty((ncb, A, P_, nx), dtype=torch.complex64,
                                device=self.torch_dev)
                for k1, k2 in batch_iterator(K, kb):
                    SWx = (mps_cd[:, None, None] * U[None, k1:k2, None]
                           * alphas[None, None, :]).contiguous()
                    Fk = self._fft2(SWx).reshape(ncb, k2 - k1, nsub, nx, P_).transpose(-1, -2)
                    # a = l*K + k, so scatter this k-slice into every l block
                    for l in range(nsub):
                        F[:, l * K + k1: l * K + k2] = Fk[:, :, l]
                    del SWx, Fk

                _apply_dedup_gram(gu_l[i], perm_l[i], grp_l[i], F)   # in place

                for k1, k2 in batch_iterator(K, kb):
                    Fk = torch.empty((ncb, k2 - k1, nsub, P_, nx),
                                     dtype=torch.complex64, device=self.torch_dev)
                    for l in range(nsub):
                        Fk[:, :, l] = F[:, l * K + k1: l * K + k2]
                    Z = self._ifft2(Fk.transpose(-1, -2).reshape(ncb, k2 - k1, nsub, nx, ny, nz))
                    out += einsum(Z, mps_cd.conj(), U[k1:k2].conj(),
                                  'nc lk lsub nx ny nz, nc nx ny nz,'
                                  ' lk nx ny nz -> lsub nx ny nz')
                    del Fk, Z
                del F, U
        return out


class subspace_cart_B0_worder_linop(linop):
    # B0 wrong order

    def __init__(self,
                 im_size: tuple,
                 mask_sample: torch.Tensor,
                 mps: torch.Tensor,
                 phi: torch.Tensor,
                 motion_bin: torch.Tensor,
                 dcf: Optional[torch.Tensor] = None,
                 imperf_model: Optional[tuple] = None,
                 imperf_model_motion: Optional[imperfection] = None,
                 use_toeplitz: Optional[bool] = False,
                 bparams: Optional[batching_params] = batching_params()):
        """
        Parameters
        ----------
        im_size : tuple 
            image dims as tuple of ints (dim1, dim2, ...)
        mask_sample : torch.tensor <float> | GPU
            The k-space trajectory with shape (npe, ntr, nro, d). 
                we assume that trj values are in [-n/2, n/2] (for nxn grid)
        mps : torch.tensor <complex> | GPU
            sensititvity maps with shape (ncoil, ndim1, ..., ndimN)
        phi : torch.tensor <complex> | GPU
            subspace basis with shape (nsub, ntr)   
        motion_bin : torch.tensor <int> | GPU
            which pe belongs to which motion states (1,npe)
        dcf : torch.tensor <float> | GPU
            the density comp. functon with shape (nro, ...)
        nufft : NUFFT
            the nufft object, defaults to torchkbnufft
        imperf_model : a struct cattying B0 and TEs
            
        imperf_model_motion : motion operator
        use_toeplitz : bool
            toggles toeplitz normal operator
        bparams : batching_params
            contains the batch sizes for the coils, subspace coeffs, and field segments
        """
        
        ishape = (phi.shape[0], *im_size)
        oshape = (mps.shape[0], motion_bin.shape[0],mps.shape[-1])
        super().__init__(ishape, oshape)

        # Consts
        torch_dev = mps.device
        assert phi.device == torch_dev
        assert mps.device == torch_dev

        # Default params
        # if nufft is None:
        #     nufft = torchkb_nufft(im_size, torch_dev.index)
        if dcf is None:
            dcf = torch.ones((motion_bin.shape[0],mps.shape[-1]), dtype=torch.float32, device=torch_dev)
        else:
            assert dcf.device == torch_dev

        if imperf_model is None:
            b0 = torch.zeros((1,*mps.shape), dtype=torch.float32, device=torch_dev)
            tes = torch.zeros((phi.shape[1]), dtype=torch.float32, device=torch_dev)
            phs0 = torch.zeros((1,*mps.shape), dtype=torch.float32, device=torch_dev)
            phst = torch.zeros((phi.shape[1]), dtype=torch.float32, device=torch_dev)
        else:
            b0 = imperf_model[0]
            tes = imperf_model[1]
            phs0 = imperf_model[2]
            phst = imperf_model[3]
            
        # Rescale and type cast
        mask_sample = mask_sample.type(torch.float32)
        mask_sample = mask_sample>0
        num_sample = motion_bin.shape[0]
        dcf = dcf.type(torch.float32)
        mps = mps.type(torch.complex64)
        phi = phi.type(torch.complex64)
        b0 = b0.type(torch.complex64)
        tes = tes.type(torch.complex64)
        phs0 = phs0.type(torch.complex64)
        phst = phst.type(torch.complex64)
        motion_bin = motion_bin.type(torch.int)
        nt = phi.shape[1]
        nc = mps.shape[0]
        nx = mps.shape[-1]
        nz = mps.shape[-2]
        ny = mps.shape[-3]
        nmot = b0.shape[0]
        
        # Compute toeplitz kernels
        self.toep_kerns = None
        
        # Save
        self.im_size = im_size
        self.use_toeplitz = use_toeplitz
        self.mask_sample = mask_sample
        self.num_sample = num_sample
        self.phi = phi
        self.mps = mps
        self.dcf = dcf
        self.tes = tes
        self.b0 = b0
        self.phs0 = phs0
        self.phst = phst
        self.motion_bin = motion_bin
        self.imperf_model_motion = imperf_model_motion
        # self.nm = phs.shape[0]
        # self.nt = phi.shape[1]
        self.bparams = bparams
        self.torch_dev = torch_dev

    def forward(self,
                alphas: torch.Tensor) -> torch.Tensor:
       
        # Useful constants
        nt = self.phi.shape[1]
        nc = self.mps.shape[0]
        nx = self.mps.shape[-1]
        nz = self.mps.shape[-2]
        ny = self.mps.shape[-3]
        nmot = self.b0.shape[0]
        im_size = self.im_size
        # print(f'im_size {im_size}')
        num_sample = self.num_sample
        motion_bin = self.motion_bin
        coil_batch_size = self.bparams.coil_batch_size
        # sub_batch_size = self.bparams.sub_batch_size
        seg_batch_size = self.bparams.field_batch_size
        # motion_batch_size = self.bparams.motion_batch_size

        # Result array
        ksp = torch.zeros((nc, num_sample,nx), dtype=torch.complex64, device=self.torch_dev)

        # Batch over coils
        for c, d in batch_iterator(nc, coil_batch_size):
            # mps = self.mps[c:d]
            # Batch over time segments
            # for a, b in batch_iterator(nmot, motion_batch_size):

            for l1, l2 in batch_iterator(nt, seg_batch_size):
                len_t = l2-l1
                # FSTBPx_t = torch.zeros((nmot, d-c, len_t, ny,nz,nx), dtype=torch.complex64, device=self.torch_dev)
                # phi
                Px = einsum(alphas,self.phi[:,l1:l2], 'nsub nro npe nx, nsub nt -> nt nro npe nx')
                # phs 
                # phs = torch.exp(1j *  (2 * torch.pi * einsum(self.b0, self.tes[l1:l2],'nmot nro npe nx, nt -> nmot nt nro npe nx') + einsum(self.phs0, self.phst[l1:l2],'nmot nro npe nx, nt -> nmot nt nro npe nx')))
                
                # BPx = einsum(Px,phs, 'nt nro npe nx, nmot nt nro npe nx -> nmot nt nro npe nx')
                # BPx = einsum('ikjm->aikjm',Px).expand(nmot, -1,-1,-1,-1)
                # print(f'PX size{Px.shape}')
                BPx = einsum(torch.ones(nmot,dtype=torch.complex64).to(self.torch_dev),Px,'nmot, nt nro npe nx -> nmot nt nro npe nx')
                
                # print(f'BPX size{BPx.shape}')
                # motion correction
                if self.imperf_model_motion is None:
                    TPx = BPx
                else:
                    TPx = self.imperf_model_motion.apply_spatial(BPx)
                          
                
                phs1 =  2 * torch.pi * einsum(self.b0, self.tes[l1:l2],'nmot nro npe ntr, nt -> nmot nt nro npe ntr')
                phs2 = einsum(self.phs0, self.phst[l1:l2],'nmot nro npe ntr, nt -> nmot nt nro npe ntr')
                phs =  torch.exp(1j*(phs1+phs2))
                BTPx = einsum(TPx,phs, 'nmot nt nro npe nx, nmot nt nro npe nx -> nmot nt nro npe nx')
                # sensitivity
                STBPx = einsum(BTPx, self.mps[c:d,...], 'nmot nt nro npe nx, nc nro npe nx -> nmot nc nt  nro npe nx')

                FSTBPx = fftc(STBPx, dim=(-3,-2))
            
                # FSTBPx_t[:,:,l1:l2,...] = FSTBPx
                for i in range(nmot):
                    # Append to k-space
                    temp = FSTBPx[i,...].reshape(d-c,len_t*ny*nz,nx)
                    st = self.mask_sample[i,:l1,...].squeeze().sum().int()
                    ed = self.mask_sample[i,:l2,...].squeeze().sum().int()
                    tmp = ksp[c:d,motion_bin==i,...] 
                    tmp[:,st:ed,...] = temp[:,self.mask_sample[i,l1:l2,...].flatten()==1,:]
                    ksp[c:d, motion_bin==i,:] = tmp
                
            

        return ksp
    
    def adjoint(self,
                ksp: torch.Tensor) -> torch.Tensor:

        # Useful constants
        nt = self.phi.shape[1]
        nsub = self.phi.shape[0]
        nc = self.mps.shape[0]
        nx = self.mps.shape[-1]
        nz = self.mps.shape[-2]
        ny = self.mps.shape[-3]
        nmot = self.b0.shape[0]
        motion_bin = self.motion_bin
        coil_batch_size = self.bparams.coil_batch_size
        # sub_batch_size = self.bparams.sub_batch_size
        seg_batch_size = self.bparams.field_batch_size
        # motion_batch_size = self.bparams.motion_batch_size
        # Result subspace coefficients
        alphas = torch.zeros((nsub, *self.im_size), dtype=torch.complex64, device=self.torch_dev)  
        # Batch over coils
        for c, d in batch_iterator(nc, coil_batch_size):
            mps = self.mps[c:d]

            for l1, l2 in batch_iterator(nt, seg_batch_size):   
                len_t = l2-l1        
                SFWy = torch.zeros((nmot, len_t, ny, nz,nx), dtype=torch.complex64, device=self.torch_dev)
                for i in range (nmot):
                    ksp_weighted = torch.zeros((d-c,len_t*ny*nz,nx), dtype=torch.complex64, device=self.torch_dev)
                    
                    st = self.mask_sample[i,:l1,...].squeeze().sum().int()
                    ed = self.mask_sample[i,:l2,...].squeeze().sum().int()

                    tmp = ksp[c:d,motion_bin==i,...] 

                    ksp_weighted[:,self.mask_sample[i,l1:l2,...].flatten()==1,:] = tmp[:,st:ed,...] 
                    
                    ksp_weighted = ksp_weighted.reshape(d-c,len_t,ny,nz,nx)
                                
                    FWy = ifftc(ksp_weighted, dim=(-3,-2)) # nc nsub nseg *im_size
                    # Conjugate maps
                    SFWy[i,...] = einsum(FWy, mps.conj(), 'nc nt ..., nc ... -> nt ...')
                # Conjugate imperfection maps
                phs1 =  2 * torch.pi * einsum(self.b0, self.tes[l1:l2],'nmot nro npe ntr, nt -> nmot nt nro npe ntr')
                phs2 = einsum(self.phs0, self.phst[l1:l2],'nmot nro npe ntr, nt -> nmot nt nro npe ntr')
                phs =  torch.exp(1j*(phs1+phs2))
                BSFWy = einsum(SFWy,phs.conj(),'nmot nt ..., nmot nt ... -> nmot nt ...')
                
                if self.imperf_model_motion is None:
                    TBSFWy = BSFWy
                else:
                    TBSFWy = self.imperf_model_motion.apply_spatial_adjoint(BSFWy)
                
                TBSFWy = torch.sum(TBSFWy,dim=0)
                PTBSFWy = einsum(TBSFWy,self.phi[:,l1:l2], 'nt nro npe ntr, nsub nt -> nsub nro npe ntr')
                    
                # Append to image
                alphas+= PTBSFWy

        return alphas
    
    def normal(self,
               alphas: torch.Tensor) -> torch.Tensor:
        return self.adjoint(self.forward(alphas))


# NW, 20260904, spatial-subspace ("spatial compress") recon, 3D, multi-B0-map.
# Python port of the MATLAB Funcs/recon/spatial_subspace_compress*.m operator; M
# and Phi_all are built by EPTI_fft_spatial.py's build_spatial_subspace().
# ── The spatial-subspace linop ───────────────────────────────────────────────
# Forward model, per B0 map / bin i:
#
#     ksp_i = Mask_i . F_yz . S . exp(1j*(2pi*db0_i*tes + phs0_i*phst)) . Phi_all . M . alpha
#             \_____/   \___/   \_/  \_______________________________/   \_____/   \_/
#             sample    fft2    mps         per-bin residual phase        long     per-voxel
#             + bin    (ny,nz)                (nmot axis)                 basis    compression
#
# i.e. exactly subspace_cart_linop with the per-voxel compression M prepended,
# and with b0 = the RESIDUAL B0 of each bin (the bulk of B0 lives inside M /
# Phi_all, which is the whole point of the spatial subspace).
#
# Differences from the notebook's subspace_cart_sp_linop / subspace_cart_or_linop
# that matter:
#   * the phase term is actually applied (the notebook comments it out in the
#     `sp` class, which is what makes it single-B0 only);
#   * the adjoint contracts with conj(phi), not phi. forward() expands via a
#     plain unconjugated `einsum(alphas, phi)`, so the Hermitian adjoint must
#     conjugate -- both notebook classes use bare `self.phi` and therefore fail
#     a dot-product adjoint test whenever phi has an imaginary part, which
#     Phi_all always does (see the same note in subspace_cart_linop.adjoint);
#   * M is applied once per forward/adjoint call rather than once per
#     (coil batch x time batch). M is the big tensor here -- (L_l, L_s, *im_size)
#     -- so hoisting it out of the loops is the difference between touching a
#     ~GB tensor twice and touching it a few hundred times. Mathematically
#     identical: M does not depend on coil or echo.
#
# TWO OPERATOR ORDERINGS (self.fast). S is a voxel-wise multiply and F acts on
# space, while Phi_all acts on the echo/subspace index, so
#
#     F[S . Phi_all . Wx](t) = sum_l Phi_all[l,t] F[S . Wx_l]
#
# i.e. S and F commute with the Phi contraction. Applying them to the L_l
# coefficient maps and contracting Phi afterwards costs nc*L_l FFTs instead of
# nc*nt. This is the ordering the notebook's subspace_cart_sp_linop used (fftc
# before the phi einsum), and it is what self.fast = True selects.
#
# HOW MUCH IT ACTUALLY BUYS -- measured, not predicted. The FFT count really does
# drop (48 -> 23 for L_l = 23), but the reordering also moves the Phi->echo
# expansion BEHIND the coil axis, so that einsum costs nc*nt*L_l per voxel
# instead of nt*L_l. At ~L_l complex MACs per element the expansion is more
# expensive per element than the 2D FFT it displaces, so the trade only pays when
# nc is small or the FFT calls are small enough that per-call overhead dominates:
#     24x20,  nc=6,  100% sampled : 1.19x fwd / 1.32x adj
#    240x216, nc=4,  100% sampled : 1.46x fwd / 1.07x adj
#    240x216, nc=4,    2% sampled : 1.14x fwd / 0.68x adj
#    240x216, nc=20,   2% sampled : 1.03x fwd / 0.70x adj   <- production shape
# So at production size it is a wash to a slight loss, and the earlier
# small-phantom numbers were an artifact of FFT call overhead.
#
# What would actually make it (and ghost_model='exact_tes_fast') pay: expand to
# echo space ONLY at the sampled points. FSWx is already in the L_l basis, and
# the mask keeps ~2% of the (nt,ny,nz) grid, so gathering the sampled (t,y,z)
# and contracting phi there costs nc*nsamp*L_l instead of nc*nt*ny*nz*L_l -- a
# ~48x cut that would leave the FFT count as the dominant term. Not implemented.
#
# It is only valid when nothing voxel-wise AND echo-dependent sits between Phi
# and F, hence the gate `(not use_phase) and imperf_model_motion is None`. With
# a per-bin residual B0 phase or a motion operator, Phi has to be contracted
# first (the standard ordering).
#
# No low-rank B0-phase model (exact_tes / combined_exact_tes, as used by
# subspace_cart_linop_fastb0) is needed or wanted here: B0 is carried entirely by
# the higher-rank Phi_all and the per-voxel M, which is the whole point of the
# spatial subspace. The phase slot only ever holds the bin-to-bin B0 residual
# (zero for a single B0 map) and, if enabled, the odd/even ghost term. That ghost
# term is exactly rank 2 in time (phst is 0/1), so should it be needed later it
# can be handled as two echo groups at 2*L_l FFTs -- worth doing only when
# nt > 2*L_l, e.g. the nt = 165 protocols, not at nt = 48.
class subspace_cart_sp_linop(linop):

    def __init__(self,
                 im_size: tuple,
                 mask_sample: torch.Tensor,
                 mps: torch.Tensor,
                 phi: torch.Tensor,
                 motion_bin: torch.Tensor,
                 dcf: Optional[torch.Tensor] = None,
                 imperf_model: Optional[tuple] = None,
                 imperf_model_motion: Optional[imperfection] = None,
                 ghost_model: Optional[str] = 'exact',
                 fast_ordering: Optional[str] = 'auto',
                 gram_normal: Optional[str] = 'auto',
                 gram_max_gb: Optional[float] = 8.0,
                 gram_store: Optional[str] = 'auto',
                 use_toeplitz: Optional[bool] = False,
                 bparams: Optional[batching_params] = batching_params()):
        """
        Parameters
        ----------
        im_size : tuple
            image dims as tuple of ints (ny, nz, nx)
        mask_sample : torch.tensor <bool> | GPU
            sampling mask with shape (nmot, nt, ny, nz)
        mps : torch.tensor <complex> | GPU
            sensitivity maps with shape (ncoil, ny, nz, nx)
        phi : torch.tensor <complex> | GPU
            the LONG subspace basis Phi_all with shape (L_l, nt)
        motion_bin : torch.tensor <int> | GPU
            which sample belongs to which B0 map / motion state, shape (num_sample,)
        dcf : torch.tensor <float> | GPU
            density compensation, shape (num_sample, nx)
        imperf_model : tuple
            (b0, tes, phs0, phst, M) where
              b0   : (nmot, ny, nz, nx)   RESIDUAL B0 of each bin, in Hz
              tes  : (nt,)                echo times, in s
              phs0 : (nmot, ny, nz, nx)   odd/even (ghost) phase
              phst : (nt,)                its per-echo on/off pattern
              M    : the per-voxel compression, either
                     - (L_l, L_s, ny, nz, nx) complex, the materialised matrix, or
                     - (dic_w_flat, pair_idx) from
                       motion_est.spatial_subspace.build_pair_index, i.e. a
                       (n_pair, L_s, L_l) table plus one index per voxel. M is
                       then gathered in voxel chunks and never materialised --
                       6 MB + 50 MB instead of 20.6 GB for a 1 mm volume, which
                       is what makes a whole-volume high-res recon fit. Same
                       arithmetic either way (verified bit-identical).
        imperf_model_motion : motion operator
        ghost_model : str
            how the odd/even (ghost) phase exp(1j*phs0*phst[t]) is applied. Same
            spirit as B0_MODEL in EPTI_fft_recon_moco.py:
              'exact'          -- built densely at every echo, (nmot, nt, *im_size),
                                  and multiplied in echo space. Forces the standard
                                  Phi-before-F ordering, i.e. nt FFTs per coil.
              'exact_tes_fast' -- phst takes only G distinct values (G = 2 for the
                                  usual 0/1 odd/even pattern), so the ghost phase is
                                  EXACTLY rank G in time. The echoes are grouped by
                                  phst value and one fixed spatial map
                                  exp(1j*phs0*v) is applied per group, which lets S
                                  and F move in front of the Phi contraction:
                                  G*L_l FFTs per coil instead of nt. Exact, not an
                                  approximation. Needs the residual B0 phase to be
                                  zero -- here B0 is carried by Phi_all/M, so that
                                  holds for a single B0 map -- and no motion
                                  operator; it falls back to 'exact' otherwise.
                                  A win only when nt > G*L_l: at nt = 48, L_l = 23
                                  that is 46 vs 48 (nothing), at nt = 165 it is
                                  46 vs 165 (3.6x).
        fast_ordering : str
            which operator ordering to use, 'auto' | 'on' | 'off'.
              'off'  -- always contract Phi first (standard). The tensor going
                        into fftc is (nmot, nc_b, nt_b, *im_size).
              'on'   -- use the F-before-Phi ordering whenever it is legal.
                        fftc then sees (nmot, nc_b, L_l, *im_size), i.e. L_l/nt_b
                        times bigger -- 5x at L_l = 20, nt_b = 4 -- and fftc
                        itself makes ~4 transient copies, so this is by far the
                        larger memory footprint. It also holds n_grp of those
                        across the whole echo loop.
              'auto' -- legacy behaviour: 'on' whenever legal.
            MEASURED at production shape (240x216, nc=20, 2% sampled) the fast
            ordering is 1.03x forward and 0.70x adjoint, i.e. a net LOSS, because
            the reordering moves the Phi->echo expansion behind the coil axis.
            It only wins for small volumes/few coils or long echo trains, so
            'off' is the right default for high-res work.
        use_toeplitz : bool
            toggles toeplitz normal operator (not implemented here)
        bparams : batching_params
            batch sizes for coils / subspace coeffs / field segments
        """

        assert imperf_model is not None, 'imperf_model must carry M (slot 4)'
        Ms = imperf_model[4]
        # M either materialised, (L_l, L_s, *im_size), or as a (table, index) pair
        self.dic_w, self.pair_idx = None, None
        if isinstance(Ms, (tuple, list)):
            dic_w, pair_idx = Ms
            K_s, K_l = dic_w.shape[1], dic_w.shape[2]
        else:
            K_l, K_s = Ms.shape[0], Ms.shape[1]
        ishape = (K_s, *im_size)                                      # (L_s, ny, nz, nx)
        oshape = (mps.shape[0], motion_bin.shape[0], mps.shape[-1])   # (nc, num_sample, nx)
        super().__init__(ishape, oshape)

        # Consts
        torch_dev = mps.device
        assert phi.device == torch_dev
        assert mps.device == torch_dev

        if dcf is None:
            dcf = torch.ones((motion_bin.shape[0], mps.shape[-1]), dtype=torch.float32, device=torch_dev)
        else:
            assert dcf.device == torch_dev

        b0 = imperf_model[0]
        tes = imperf_model[1]
        phs0 = imperf_model[2]
        phst = imperf_model[3]

        # Rescale and type cast
        mask_sample = mask_sample.type(torch.float32)
        mask_sample = mask_sample > 0
        num_sample = motion_bin.shape[0]
        dcf = dcf.type(torch.float32)
        mps = mps.type(torch.complex64)
        phi = phi.type(torch.complex64)
        b0 = b0.type(torch.complex64)
        tes = tes.type(torch.complex64)
        phs0 = phs0.type(torch.complex64)
        phst = phst.type(torch.complex64)
        if isinstance(Ms, (tuple, list)):
            self.dic_w = Ms[0].type(torch.complex64).to(torch_dev)          # (n_pair, K_s, K_l)
            self.pair_idx = Ms[1].reshape(-1).long().to(torch_dev)          # (V,)
            assert self.pair_idx.numel() == int(np.prod(im_size)), \
                'pair_idx must have one entry per voxel'
            Ms = None
        else:
            Ms = Ms.type(torch.complex64)
            assert Ms.device == torch_dev
        # SHIFT-FREE centred FFT (this class only). For even N,
        #   fftshift(fftn(ifftshift(x)))/sqrt(N) == c * chk * fftn(chk*x, 'ortho')
        # with chk[y,z] = (-1)^(y+z) and c = (-1)^(ny/2+nz/2), verified to 2e-07.
        # The image-side chk and c are folded into mps once here, so nothing extra
        # is multiplied in the loops; only the k-space chk is applied, in place.
        # That takes the centred FFT from 3 allocations of the (often multi-GB)
        # k-space tensor down to 1. Odd ny or nz breaks the identity (fftshift !=
        # ifftshift there), so those fall back to the shifting fftc/ifftc.
        ny_, nz_ = im_size[0], im_size[1]
        self.shift_free = (ny_ % 2 == 0) and (nz_ % 2 == 0)
        if self.shift_free:
            _y = torch.arange(ny_, device=torch_dev)
            _z = torch.arange(nz_, device=torch_dev)
            chk = ((-1.0) ** ((_y[:, None] + _z[None, :]) % 2)).type(torch.float32)
            c = float((-1.0) ** (((ny_ // 2) + (nz_ // 2)) % 2))
            self.chk_k = chk[:, :, None]                      # (ny, nz, 1)
            mps = mps * (c * self.chk_k)                      # image side, once
        else:
            self.chk_k = None
            print(f'subspace_cart_sp_linop: ny={ny_} or nz={nz_} is odd -> keeping the '
                  'shifting fftc (the checkerboard identity needs even dims)')

        self.K_s, self.K_l = K_s, K_l
        # voxels per gather chunk, sized so the gathered block stays ~256 MB
        self.vox_chunk = max(1, int(256e6 // (K_s * K_l * 8)))
        motion_bin = motion_bin.type(torch.int)

        # What phase is actually present?
        b0_zero = bool(b0.abs().max().item() == 0)      # residual B0 (0 for one B0 map)
        ghost_on = bool(phs0.abs().max().item() != 0)   # odd/even term
        self.use_phase = (not b0_zero) or ghost_on
        assert ghost_model in ('exact', 'exact_tes_fast'), \
            f'unknown ghost_model {ghost_model!r}'
        self.ghost_model = ghost_model

        # The fast (F before Phi) ordering needs nothing voxel-wise AND
        # echo-dependent between Phi and F: no residual B0 phase, no motion. The
        # ghost term is allowed in 'exact_tes_fast' because it is exactly rank G
        # in time -- the echoes are grouped by phst value and each group carries
        # one fixed spatial map, which commutes with the Phi contraction.
        fast_ok = b0_zero and (imperf_model_motion is None)
        if ghost_model == 'exact_tes_fast' and not fast_ok:
            print('subspace_cart_sp_linop: ghost_model=exact_tes_fast needs a zero '
                  'residual B0 phase and no motion operator -> falling back to exact')
            self.ghost_model = ghost_model = 'exact'
        assert fast_ordering in ('auto', 'on', 'off'), \
            f'unknown fast_ordering {fast_ordering!r}'
        # NOTE ghost_model only selects an ordering when the ghost term is
        # PRESENT; with it absent both models are legal for the fast path, which
        # is why 'exact' alone never forced the standard ordering. fast_ordering
        # is the explicit control.
        self.fast = (fast_ordering != 'off') and fast_ok \
            and (ghost_model == 'exact_tes_fast' or not ghost_on)
        if ghost_model == 'exact_tes_fast' and fast_ordering == 'off':
            print('subspace_cart_sp_linop: fast_ordering=off overrides '
                  'ghost_model=exact_tes_fast (its rank-G grouping only helps the '
                  'F-before-Phi ordering); the ghost phase is still applied exactly')

        # Echo groups for the fast path: one group per distinct phst value, with
        # its spatial map exp(1j*phs0*v). Ghost off -> a single group with map 1
        # (self.ghost_maps stays None), which is the plain no-phase fast path.
        # Built whenever the ghost term is present, regardless of ordering: the
        # standard path uses them too (see _apply_phase), so no exp() is ever
        # evaluated inside the operator loops.
        self.grp_mask = None
        self.ghost_maps = None
        self.n_grp = 1
        self.b0_zero = b0_zero
        if ghost_on:
            vals = torch.unique(phst.real)
            self.grp_mask = torch.stack([(phst.real == v) for v in vals])      # (G, nt)
            self.ghost_maps = torch.exp(1j * einsum(phs0.real.type(torch.float32),
                                                    vals.type(torch.float32),
                                                    'nmot ..., g -> g nmot ...')
                                        ).type(torch.complex64)                # (G, nmot, *im_size)
            self.n_grp = int(vals.numel())

        n_fft = phi.shape[1] if not self.fast else self.n_grp * K_l
        print(f'subspace_cart_sp_linop: mask {tuple(mask_sample.shape)}, '
              f'L_s = {K_s}, L_l = {K_l}, nmot = {b0.shape[0]}, '
              f'phase {"OFF" if not self.use_phase else ("ghost only" if b0_zero else "B0 residual + ghost")}'
              f'{"" if not self.use_phase else " (" + self.ghost_model + ")"}, '
              f'{"FAST (F before Phi, %d group(s))" % self.n_grp if self.fast else "standard (Phi before F)"}'
              f', {n_fft} FFTs/coil'
              f'{", M gathered from a %d-entry table" % self.dic_w.shape[0] if self.dic_w is not None else ""}')
        V = int(np.prod(im_size))
        cb = min(bparams.coil_batch_size, mps.shape[0])
        fb = min(bparams.field_batch_size, phi.shape[1])
        if self.fast:
            fft_in = b0.shape[0] * cb * K_l * V * 8
            held = self.n_grp * fft_in
        else:
            fft_in = b0.shape[0] * cb * fb * V * 8
            held = fft_in
        print(f'  centred FFT: {"shift-free (1 alloc/call)" if self.shift_free else "shifting fftc (3 allocs/call)"}'
              f', input {fft_in / 1e9:.2f} GB, '
              f'{held / 1e9:.2f} GB held across the echo loop'
              f'{"  <-- lower it with coil_batch_size, or fast_ordering=off" if self.fast else ""}')

        # Compute toeplitz kernels
        self.toep_kerns = None

        # Save
        self.im_size = im_size
        self.use_toeplitz = use_toeplitz
        self.mask_sample = mask_sample
        self.num_sample = num_sample
        self.phi = phi
        self.mps = mps
        self.dcf = dcf
        self.tes = tes
        self.b0 = b0
        self.phs0 = phs0
        self.phst = phst
        self.M = Ms
        self.motion_bin = motion_bin
        self.imperf_model_motion = imperf_model_motion
        self.bparams = bparams
        self.torch_dev = torch_dev

        # ── sample bookkeeping, precomputed ONCE ────────────────────────────
        # The scatter/gather used to recompute, EVERY call:
        #   st/ed = mask_sample[i,:l].sum().int()     -> 0-dim CUDA tensor used as
        #                                                a slice bound = D2H sync
        #   ksp[c:d, motion_bin == i]                 -> boolean advanced index =
        #                                                nonzero() + a full copy of
        #                                                the bin's whole block
        #   mask_sample[i,l1:l2].flatten() == 1       -> another nonzero()
        # With coil_batch 2 / field_batch 2 on a 48-echo 20-coil dataset that is
        # 240 calls x ~3 syncs = ~720 pipeline drains per forward, plus ~96 GB of
        # redundant copying -- which is why one A.normal() took ~130 s on an A6000
        # instead of well under a second. All of it is static, so it is computed
        # here: host-side ints for the bounds, one device index tensor per bin.
        _msk = mask_sample.reshape(mask_sample.shape[0], mask_sample.shape[1], -1)
        _cnt = _msk.sum(-1).cpu().numpy()                        # (nmot, nt)
        self.cum = np.concatenate([np.zeros((_cnt.shape[0], 1), dtype=np.int64),
                                   np.cumsum(_cnt, axis=1)], axis=1)   # (nmot, nt+1)
        self.sel_full = [torch.nonzero(_msk[i].reshape(-1)).flatten()
                         for i in range(_msk.shape[0])]           # device, sorted
        # each bin occupies a contiguous run of the sample axis (toPython_recon.m
        # builds motion_bin as blocks of increasing bin index), which lets the
        # scatter/gather use plain slices -- no mask, no copy, no sync
        _mb = motion_bin.cpu().numpy()
        self.bin_start = np.zeros(_msk.shape[0], dtype=np.int64)
        self.contig_bins = True
        for i in range(_msk.shape[0]):
            w = np.nonzero(_mb == i)[0]
            if w.size == 0:
                continue
            self.bin_start[i] = w[0]
            if w[-1] - w[0] + 1 != w.size or w.size != self.cum[i, -1]:
                self.contig_bins = False
        if not self.contig_bins:
            print('subspace_cart_sp_linop: motion_bin is not contiguous per bin -> '
                  'falling back to boolean-mask scatter/gather (slower)')

        # ── per-sample bookkeeping for the SAMPLED-POINT expansion ──────────
        # In the F-before-Phi ordering the k-space coefficient maps are available,
        # so Phi can be contracted only where data was actually measured instead
        # of over the dense (nt, ny, nz) grid. The mask keeps ~2% of that grid --
        # every echo is sampled, but only ~1080 of 51840 (ky,kz) per echo -- so
        # the dense expansion computes and discards ~98% of its output.
        # Per (bin, echo group) this stores, in sample order:
        #   samp_idx : offset of each sample within the bin's block of ksp
        #   samp_pos : its flat (ky, nz) position, to gather from the k-space maps
        #   samp_phi : Phi[:, t] for its echo, so no per-echo indexing is needed
        P_ = int(im_size[0]) * int(im_size[1])
        self.samp_idx, self.samp_pos, self.samp_phi = [], [], []
        for i in range(_msk.shape[0]):
            full = self.sel_full[i]                       # sorted, into nt*P
            t_of = torch.div(full, P_, rounding_mode='floor')
            p_of = full - t_of * P_
            idx_i, pos_i, phi_i = [], [], []
            for g in range(self.n_grp):
                if self.grp_mask is None:
                    sel = torch.ones_like(t_of, dtype=torch.bool)
                else:
                    sel = self.grp_mask[g].to(t_of.device)[t_of]
                w = torch.nonzero(sel).flatten()
                # sorted by k-space position: the adjoint scatter-adds into that
                # axis, and monotone indices make it ~10x faster (measured
                # 118 ms vs 308 ms, and 12 ms once the axis is outermost too)
                w = w[torch.argsort(p_of[w])]
                idx_i.append(w)                           # into [0, n_i)
                pos_i.append(p_of[w].contiguous())
                phi_i.append(phi[:, t_of[w]].contiguous())   # (K_l, m)
            self.samp_idx.append(idx_i)
            self.samp_pos.append(pos_i)
            self.samp_phi.append(phi_i)
        # samples per gather chunk, sized so the gathered block stays ~256 MB
        self.samp_chunk = max(1, int(256e6 // (max(1, K_l) * mps.shape[-1] * 8)))

        # ── TEMPORAL GRAM ("Toeplitz in t") for a direct normal() ───────────
        # A = sum_{i,g} Ma_{i,g} Phi F S D_g M, and Ma_{i,g}^H Ma_{i',g'} = 0 for
        # (i,g) != (i',g') because each sample belongs to exactly one bin and one
        # echo group. So
        #     A^H A = sum_g M^H D_g^H S^H F^H [Gram_g] F S D_g M
        # where the bracket is DIAGONAL in (ky,kz): at position p it is
        #     Gram_g[p] = sum_{t in T_p, t in group g} Phi[:,t] Phi[:,t]^H
        # one L_l x L_l matrix. The echo axis therefore disappears from A^H A:
        # 2*n_grp*L_l FFTs per coil instead of 2*nt (4.3x fewer at nt = 196,
        # L_l = 23), no gather/scatter, and the nc x nsamp x nx sample-space
        # intermediate is never formed.
        # Requires no image-space per-echo phase (b0 residual zero, else the echo
        # axis lives in image space too), no motion operator, and a ghost map that
        # does not depend on the bin (true whenever phase_diff has a singleton bin
        # axis, which is what toPython_recon.m writes).
        self.gram = None
        self.gram_dedup = None
        assert gram_normal in ('auto', 'on', 'off'), \
            f'unknown gram_normal {gram_normal!r}'
        if gram_normal != 'off' and b0_zero and imperf_model_motion is None:
            bin_indep = (self.ghost_maps is None or self.ghost_maps.shape[1] == 1
                         or bool(torch.allclose(self.ghost_maps,
                                                self.ghost_maps[:, :1].expand_as(self.ghost_maps))))
            gb = self.n_grp * P_ * K_l * K_l * 8 / 1e9
            # The Gram trades 2*nt FFTs per coil for 2*n_grp*L_l FFTs plus one
            # L_l x L_l matvec per (ky,kz). That matvec is not free, so the trade
            # only wins when the FFT count actually drops a lot. Measured on real
            # data, nc = 20, 240x216x4, n_grp = 2, L_l = 23:
            #   nt =  48 -> 92 vs  96 FFTs: 59.6 s vs 49.4 s, a 1.2x LOSS
            #   nt = 196 -> 92 vs 392 FFTs: 5.0x win
            # so 'auto' requires a >= 1.33x FFT reduction and 'on' forces it.
            useful = (self.n_grp * K_l) <= 0.75 * phi.shape[1]
            if not bin_indep:
                print('subspace_cart_sp_linop: ghost map varies with the bin -> no '
                      'temporal Gram (normal() falls back to adjoint(forward))')
            elif gram_normal == 'auto' and not useful:
                print(f'subspace_cart_sp_linop: temporal Gram would only take '
                      f'{2 * self.n_grp * K_l} FFTs/coil vs {2 * phi.shape[1]} '
                      f'(n_grp={self.n_grp}, L_l={K_l}, nt={phi.shape[1]}) -> not '
                      f'worth its per-position matmul, skipping it. Force with '
                      f'gram_normal=on.')
            elif gb > gram_max_gb:
                print(f'subspace_cart_sp_linop: temporal Gram would need {gb:.2f} GB '
                      f'(> gram_max_gb={gram_max_gb}) -> not building it')
            else:
                assert gram_store in ('auto', 'dense', 'dedup'), \
                    f'unknown gram_store {gram_store!r}'
                # 'auto': dedup once the dense form is big enough for the extra
                # gather/scatter to be worth it (a 1 mm volume, or a large L_l)
                dedup = (gram_store == 'dedup'
                         or (gram_store == 'auto' and gb > 1.0))
                gram = (None if dedup else
                        torch.zeros((self.n_grp, P_, K_l, K_l),
                                    dtype=torch.complex64, device=torch_dev))
                gu_l, inv_l = [], []
                n_uniq_tot = 0
                for g in range(self.n_grp):
                    # how many times each echo is sampled at each (ky,kz), summed
                    # over ALL bins. This is a COUNT, not a boolean: A^H A =
                    # sum_i F^H Ma_i^H Ma_i F, so a (ky,kz,t) that two bins both
                    # acquire contributes twice. ORing the bins instead undercounts
                    # the overlap and gives ~5e-2 relative error at nmot = 2.
                    cnt = np.zeros((P_, phi.shape[1]), dtype=np.uint16)
                    for i in range(_msk.shape[0]):
                        pos_i = self.samp_pos[i][g].cpu().numpy()
                        t_i = (self.sel_full[i][self.samp_idx[i][g]].cpu().numpy() // P_)
                        # (pos,t) pairs are unique within one bin (mask is boolean),
                        # so a plain += is enough; the sum across bins is the count
                        cnt[pos_i, t_i] += 1
                    # only ~(n_trajectories x block size) distinct echo sets exist,
                    # so dedup and build one Gram per distinct set
                    gu, inv = _dedup_temporal_gram(phi, cnt, torch_dev)
                    n_uniq_tot += gu.shape[0]
                    gu_l.append(gu); inv_l.append(inv)
                    if not dedup:
                        gram[g] = gu[inv]
                # DENSE vs DEDUP storage. Gram_g[p] depends only on the echo set
                # T_p, and an EPTI trajectory has very few distinct sets (~50 on
                # real data vs 51840 positions), so keeping just the unique
                # matrices plus one index per position turns 0.44 GB into well
                # under a megabyte. The cost is that the apply needs the
                # positions sorted by set, i.e. one gather + one scatter of the
                # (nc, L_l, P, nx) tensor, so dense stays the default at sizes
                # where it fits (see gram_store).
                ded_gb = sum(g_.numel() for g_ in gu_l) * 8 / 1e9
                if dedup:
                    self.gram = None
                    self.gram_dedup = [(gu_l[g], *_dedup_gram_order(inv_l[g],
                                                                    gu_l[g].shape[0]))
                                       for g in range(self.n_grp)]
                else:
                    self.gram = gram
                    self.gram_dedup = None
                print(f'  temporal Gram: {n_uniq_tot} distinct echo sets across '
                      f'{self.n_grp} group(s), stored '
                      f'{"DEDUPLICATED %.2f MB (dense would be %.2f GB)" % (ded_gb * 1e3, gb) if dedup else "dense %.2f GB (%.2f MB deduplicated)" % (gb, ded_gb * 1e3)}'
                      f' -> normal() uses {2 * self.n_grp * K_l} FFTs/coil'
                      f' instead of {2 * phi.shape[1]}')

    def _fftc(self, x: torch.Tensor) -> torch.Tensor:
        """centred 2D FFT over (ny, nz). Shift-free when the dims are even: 1
        allocation instead of 3 (the image-side checkerboard is already in mps,
        and the k-space one is applied in place)."""
        if not self.shift_free:
            return fftc(x, dim=(-3, -2))
        out = torch.fft.fftn(x, dim=(-3, -2), norm='ortho')
        out *= self.chk_k
        return out

    def _ifftc(self, y: torch.Tensor) -> torch.Tensor:
        """adjoint partner of _fftc. The output-side checkerboard and c come from
        conj(mps), which already carries them, so only the input one is applied."""
        if not self.shift_free:
            return ifftc(y, dim=(-3, -2))
        return torch.fft.ifftn(y * self.chk_k, dim=(-3, -2), norm='ortho')

    def _w_for(self, alphas: torch.Tensor) -> torch.Tensor:
        """per-voxel compression: (K_s, *im_size) -> (K_l, *im_size)."""
        if self.dic_w is None:
            return einsum(alphas.type(torch.complex64), self.M,
                          'nsub1 nro npe nx, nsub2 nsub1 nro npe nx -> nsub2 nro npe nx')
        # gather M one voxel chunk at a time; the table is small enough to stay hot
        a = alphas.type(torch.complex64).reshape(self.K_s, -1)
        out = torch.empty((self.K_l, a.shape[1]), dtype=torch.complex64,
                          device=self.torch_dev)
        for v1, v2 in batch_iterator(a.shape[1], self.vox_chunk):
            blk = self.dic_w[self.pair_idx[v1:v2]]              # (n, K_s, K_l)
            out[:, v1:v2] = einsum(a[:, v1:v2], blk, 's n, n s l -> l n')
        return out.reshape(self.K_l, *self.im_size)

    def _w_adj(self, y: torch.Tensor) -> torch.Tensor:
        """adjoint of _w_for: (K_l, *im_size) -> (K_s, *im_size)."""
        if self.dic_w is None:
            return einsum(y, self.M.conj(),
                          'nsub2 nro npe nx, nsub2 nsub1 nro npe nx -> nsub1 nro npe nx')
        yv = y.reshape(self.K_l, -1)
        out = torch.empty((self.K_s, yv.shape[1]), dtype=torch.complex64,
                          device=self.torch_dev)
        for v1, v2 in batch_iterator(yv.shape[1], self.vox_chunk):
            blk = self.dic_w[self.pair_idx[v1:v2]].conj()       # (n, K_s, K_l)
            out[:, v1:v2] = einsum(yv[:, v1:v2], blk, 'l n, n s l -> s n')
        return out.reshape(self.K_s, *self.im_size)

    def _scatter(self, ksp, src, c, d, l1, l2, nmot, ny, nz, nx, len_t):
        """Write one (coil batch, echo batch) block of k-space images into the
        compacted sample axis, per bin. src is (nmot_or_1, nc_b, nt_b, *im_size)."""
        for i in range(nmot):
            s = src[i, ...] if src.shape[0] > 1 else src[0, ...]
            temp = s.reshape(d - c, len_t * ny * nz, nx)
            st, ed = int(self.cum[i, l1]), int(self.cum[i, l2])       # host ints
            if ed == st:
                continue
            sel = self.sel_full[i][st:ed] - l1 * ny * nz             # precomputed
            if self.contig_bins:
                b0 = int(self.bin_start[i])
                ksp[c:d, b0 + st:b0 + ed, :] = temp[:, sel, :]       # direct write
            else:
                tmp = ksp[c:d, self.motion_bin == i, ...]
                tmp[:, st:ed, ...] = temp[:, sel, :]
                ksp[c:d, self.motion_bin == i, :] = tmp

    def _gather(self, ksp, i, c, d, l1, l2, ny, nz, nx, len_t):
        """Inverse of _scatter for one bin: pull that bin's samples for echo batch
        [l1:l2) back onto the dense (nc_b, nt_b, ny, nz, nx) grid."""
        ksp_weighted = torch.zeros((d - c, len_t * ny * nz, nx),
                                   dtype=torch.complex64, device=self.torch_dev)
        st, ed = int(self.cum[i, l1]), int(self.cum[i, l2])           # host ints
        if ed > st:
            sel = self.sel_full[i][st:ed] - l1 * ny * nz             # precomputed
            if self.contig_bins:
                b0 = int(self.bin_start[i])
                ksp_weighted[:, sel, :] = ksp[c:d, b0 + st:b0 + ed, :]   # a view
            else:
                tmp = ksp[c:d, self.motion_bin == i, ...]
                ksp_weighted[:, sel, :] = tmp[:, st:ed, ...]
        return ksp_weighted.reshape(d - c, len_t, ny, nz, nx)

    def _echo_groups(self, l1: int, l2: int):
        """[(group, local echo indices)] for echo batch [l1:l2)."""
        if self.grp_mask is None:
            return [(0, torch.arange(l2 - l1, device=self.torch_dev))]
        out = []
        for g in range(self.n_grp):
            gt = torch.nonzero(self.grp_mask[g, l1:l2]).flatten()
            if gt.numel():
                out.append((g, gt))
        return out

    def _apply_phase(self, Px: torch.Tensor, l1: int, l2: int) -> torch.Tensor:
        """(nt_b, *im_size) -> (nmot, nt_b, *im_size), phase applied.

        The ghost term takes only n_grp distinct values in time (phst), so its
        spatial maps exp(1j*phs0*v) are precomputed in __init__ and applied by
        broadcasting over each echo group -- no transcendental in the loop. The
        residual-B0 term is genuinely per-echo, so it is built here, but only
        when nonzero (it is identically zero for a single B0 map).
        """
        if self.b0_zero and self.ghost_maps is None:
            return Px[None, ...]                                  # no phase at all
        nmot = self.b0.shape[0]
        out = torch.empty((nmot, l2 - l1, *Px.shape[1:]),
                          dtype=torch.complex64, device=self.torch_dev)
        b0p = None
        if not self.b0_zero:
            b0p = torch.exp(1j * 2 * torch.pi
                            * einsum(self.b0, self.tes[l1:l2],
                                     'nmot nro npe ntr, nt -> nmot nt nro npe ntr'))
        for g, gt in self._echo_groups(l1, l2):
            blk = Px[gt][None, ...]                               # (1, n_g, *im)
            if self.ghost_maps is not None:
                blk = blk * self.ghost_maps[g][:, None, ...]      # (nmot, n_g, *im)
            if b0p is not None:
                blk = blk * b0p[:, gt]
            out[:, gt] = blk
        return out

    def _apply_phase_adj(self, y: torch.Tensor, l1: int, l2: int) -> torch.Tensor:
        """adjoint of _apply_phase: (nmot, nt_b, *im_size) -> (nt_b, *im_size)."""
        if self.b0_zero and self.ghost_maps is None:
            return y.sum(dim=0)
        out = torch.empty((l2 - l1, *y.shape[2:]), dtype=torch.complex64,
                          device=self.torch_dev)
        b0p = None
        if not self.b0_zero:
            b0p = torch.exp(1j * 2 * torch.pi
                            * einsum(self.b0, self.tes[l1:l2],
                                     'nmot nro npe ntr, nt -> nmot nt nro npe ntr'))
        for g, gt in self._echo_groups(l1, l2):
            blk = y[:, gt]
            if self.ghost_maps is not None:
                blk = blk * self.ghost_maps[g][:, None, ...].conj()
            if b0p is not None:
                blk = blk * b0p[:, gt].conj()
            out[gt] = blk.sum(dim=0)
        return out

    def _phase(self, l1: int, l2: int) -> torch.Tensor:
        """exp(1j*(2pi*b0*tes + phs0*phst)) for echoes [l1:l2] -- (nmot, nt, ny, nz, nx)."""
        phs1 = 2 * torch.pi * einsum(self.b0, self.tes[l1:l2],
                                     'nmot nro npe ntr, nt -> nmot nt nro npe ntr')
        phs2 = einsum(self.phs0, self.phst[l1:l2],
                      'nmot nro npe ntr, nt -> nmot nt nro npe ntr')
        return torch.exp(1j * (phs1 + phs2))

    def forward(self,
                alphas: torch.Tensor) -> torch.Tensor:

        # Useful constants
        nt = self.phi.shape[1]
        nc = self.mps.shape[0]
        nx = self.mps.shape[-1]
        nz = self.mps.shape[-2]
        ny = self.mps.shape[-3]
        nmot = self.b0.shape[0]
        num_sample = self.num_sample
        motion_bin = self.motion_bin
        coil_batch_size = self.bparams.coil_batch_size
        seg_batch_size = self.bparams.field_batch_size

        # Result array
        ksp = torch.zeros((nc, num_sample, nx), dtype=torch.complex64, device=self.torch_dev)

        # Per-voxel compression: (L_s, *im_size) -> (L_l, *im_size). Independent of
        # coil and echo, so done once.
        Wx = self._w_for(alphas)

        if not self.fast:
            # STANDARD ordering: Px, the phase and the motion operator are all
            # coil-INDEPENDENT, so the echo loop goes outside and they are
            # evaluated once per echo batch rather than once per (coil, echo)
            # batch -- at 1 mm with 5 coil batches that removes 5x of the Phi
            # expansion, the phase and the motion application.
            for l1, l2 in batch_iterator(nt, seg_batch_size):
                len_t = l2 - l1
                Px = einsum(Wx, self.phi[:, l1:l2],
                            'nsub nro npe nx, nsub nt -> nt nro npe nx')
                BPx = self._apply_phase(Px, l1, l2)          # (nmot, nt_b, *im)
                if self.imperf_model_motion is not None:
                    BPx = self.imperf_model_motion.apply_spatial(BPx)
                for c, d in batch_iterator(nc, coil_batch_size):
                    STBPx = einsum(BPx, self.mps[c:d, ...],
                                   'nmot nt nro npe nx, nc nro npe nx'
                                   ' -> nmot nc nt nro npe nx')
                    FSTBPx = self._fftc(STBPx)
                    self._scatter(ksp, FSTBPx, c, d, l1, l2, nmot, ny, nz, nx, len_t)
                del Px, BPx
            return ksp

        # FAST ordering with the sampled-point expansion. S (and the group's
        # fixed ghost map) go on the L_l coefficient maps, those are transformed
        # -- n_grp*L_l FFTs per coil instead of nt -- and then Phi is contracted
        # ONLY at the measured (t, ky, kz), so the dense (nt, ny, nz) echo grid is
        # never built. Each group is consumed before the next is formed, so only
        # one set of k-space maps is resident (the old version held all n_grp
        # across the echo loop).
        if self.fast:
            P_ = ny * nz
            for c, d in batch_iterator(nc, coil_batch_size):
                for g in range(self.n_grp):
                    if self.ghost_maps is None:
                        SWx = einsum(Wx, self.mps[c:d, ...],
                                     'nsub nro npe nx, nc nro npe nx -> nc nsub nro npe nx')[None]
                    else:
                        SWx = einsum(Wx, self.mps[c:d, ...], self.ghost_maps[g],
                                     'nsub nro npe nx, nc nro npe nx, nmot nro npe nx'
                                     ' -> nmot nc nsub nro npe nx')
                    F = self._fftc(SWx)
                    del SWx
                    F = F.reshape(F.shape[0], d - c, self.K_l, P_, nx)
                    for i in range(nmot):
                        i_f = i if F.shape[0] > 1 else 0
                        b0 = int(self.bin_start[i])
                        idx_g, pos_g, phi_g = (self.samp_idx[i][g], self.samp_pos[i][g],
                                               self.samp_phi[i][g])
                        for s1, s2 in batch_iterator(idx_g.numel(), self.samp_chunk):
                            G = F[i_f, :, :, pos_g[s1:s2], :]        # (nc_b, L_l, m, nx)
                            out = einsum(G, phi_g[:, s1:s2],
                                         'nc nsub m nx, nsub m -> nc m nx')
                            ksp[c:d, b0 + idx_g[s1:s2], :] = out
                    del F
            return ksp

        raise RuntimeError('unreachable: both orderings return above')

    def adjoint(self,
                ksp: torch.Tensor) -> torch.Tensor:

        # Useful constants
        nt = self.phi.shape[1]
        nsub2 = self.phi.shape[0]      # L_l
        nsub1 = self.K_s               # L_s
        nc = self.mps.shape[0]
        nx = self.mps.shape[-1]
        nz = self.mps.shape[-2]
        ny = self.mps.shape[-3]
        nmot = self.b0.shape[0]
        motion_bin = self.motion_bin
        coil_batch_size = self.bparams.coil_batch_size
        seg_batch_size = self.bparams.field_batch_size

        # Accumulate in the LONG basis, then apply M^H once at the end (M^H is the
        # last operator of the adjoint chain and does not depend on coil/echo).
        alphas_l = torch.zeros((nsub2, *self.im_size), dtype=torch.complex64, device=self.torch_dev)

        if not self.fast:
            # STANDARD ordering, echo loop outside: the phase adjoint, the Phi
            # contraction and the motion adjoint are coil-independent, so they run
            # once per echo batch. Only F^H and S^H are inside the coil loop.
            for l1, l2 in batch_iterator(nt, seg_batch_size):
                len_t = l2 - l1
                SFWy = torch.zeros((nmot, len_t, ny, nz, nx),
                                   dtype=torch.complex64, device=self.torch_dev)
                for c, d in batch_iterator(nc, coil_batch_size):
                    mps_c = self.mps[c:d]
                    for i in range(nmot):
                        FWy = self._ifftc(
                            self._gather(ksp, i, c, d, l1, l2, ny, nz, nx, len_t))
                        SFWy[i, ...] += einsum(FWy, mps_c.conj(),
                                               'nc nt ..., nc ... -> nt ...')
                if self.imperf_model_motion is not None:
                    SFWy = self.imperf_model_motion.apply_spatial_adjoint(SFWy)
                BTSFWy = self._apply_phase_adj(SFWy, l1, l2)      # (nt_b, *im)
                # NOTE: conj(phi) -- see the class docstring note.
                alphas_l += einsum(BTSFWy, self.phi[:, l1:l2].conj(),
                                   'nt nro npe ntr, nsub nt -> nsub nro npe ntr')
                del SFWy, BTSFWy
            return self._w_adj(alphas_l)

        # FAST ordering, mirroring the forward: the measured samples are
        # scatter-ADDED (several echoes can share a (ky,kz), so it must accumulate)
        # into the k-space coefficient maps, weighted by conj(Phi[:, t]); only then
        # is F^H applied. No dense (nt, ny, nz) grid is built, and one group is
        # finished before the next starts.
        P_ = ny * nz
        for c, d in batch_iterator(nc, coil_batch_size):
            mps = self.mps[c:d]
            n_mp = 1 if self.ghost_maps is None else nmot
            for g in range(self.n_grp):
                # the scattered (ky,kz) axis is kept OUTERMOST so each index
                # writes one contiguous (nc_b, L_l, nx) block; with the sample
                # lists pre-sorted by position that makes the scatter-add ~26x
                # cheaper than the natural (nc_b, L_l, P, nx) layout
                Py = torch.zeros((n_mp, P_, d - c, nsub2, nx),
                                 dtype=torch.complex64, device=self.torch_dev)
                for i in range(nmot):
                    i_p = i if self.ghost_maps is not None else 0
                    b0 = int(self.bin_start[i])
                    idx_g, pos_g, phi_g = (self.samp_idx[i][g], self.samp_pos[i][g],
                                           self.samp_phi[i][g])
                    for s1, s2 in batch_iterator(idx_g.numel(), self.samp_chunk):
                        blk = ksp[c:d, b0 + idx_g[s1:s2], :]           # (nc_b, m, nx)
                        # NOTE: conj(phi) -- see the class docstring note.
                        contrib = (blk.permute(1, 0, 2)[:, :, None, :]
                                   * phi_g[:, s1:s2].conj().T[:, None, :, None])
                        Py[i_p].index_add_(0, pos_g[s1:s2], contrib)
                FPy = self._ifftc(
                    Py.permute(0, 2, 3, 1, 4).reshape(n_mp, d - c, nsub2, ny, nz, nx))
                del Py
                if self.ghost_maps is None:
                    alphas_l += einsum(FPy[0], mps.conj(),
                                       'nc nsub ..., nc ... -> nsub ...')
                else:
                    alphas_l += einsum(FPy, mps.conj(), self.ghost_maps[g].conj(),
                                       'nmot nc nsub ..., nc ..., nmot ... -> nsub ...')
                del FPy

        # Per-voxel compression, adjoint
        return self._w_adj(alphas_l)

    def normal(self,
               alphas: torch.Tensor) -> torch.Tensor:
        """A^H A. With the temporal Gram this is computed directly, never
        expanding to echoes and never touching sample space:
            A^H A = sum_g M^H D_g^H S^H F^H Gram_g F S D_g M
        Gram_g being diagonal in (ky,kz) -- one L_l x L_l matrix per position. So
        2*n_grp*L_l FFTs per coil instead of the 2*nt that adjoint(forward())
        costs, no gather/scatter, and no nc x nsamp x nx intermediate. Falls back
        to adjoint(forward()) when the Gram was not built (see __init__)."""
        if self.gram is None and self.gram_dedup is None:
            return self.adjoint(self.forward(alphas))

        nc = self.mps.shape[0]
        nx, nz, ny = self.mps.shape[-1], self.mps.shape[-2], self.mps.shape[-3]
        P_ = ny * nz
        Wx = self._w_for(alphas)                          # (L_l, *im_size)
        out_l = torch.zeros((self.K_l, *self.im_size), dtype=torch.complex64,
                            device=self.torch_dev)
        for c, d in batch_iterator(nc, self.bparams.coil_batch_size):
            mps_c = self.mps[c:d]
            for g in range(self.n_grp):
                # NOTE the .contiguous() calls below are not cosmetic. einsum (and
                # a broadcast multiply against the strided Wx) returns a permuted
                # *view*; the following .reshape then stays a view too, so fftn
                # would run on a strided tensor. Measured at (240,216,2), L_l=23,
                # coil_batch 2: a strided fftn costs 388 ms vs 15 ms contiguous,
                # and the bad strides also slow the Gram matmul (278 vs 13 ms).
                # Forcing the copy makes this whole loop body 57 ms instead of
                # 336 ms, bit-for-bit identical output.
                SD = mps_c if self.ghost_maps is None else mps_c * self.ghost_maps[g, 0]
                SWx = (Wx[None] * SD[:, None]).contiguous()
                F = self._fftc(SWx)
                del SWx
                # per-position L_l x L_l matvec: a batched matmul over (ky,kz)
                if self.gram_dedup is None:
                    F = einsum(self.gram[g], F.reshape(d - c, self.K_l, P_, nx),
                               'p l m, nc m p x -> nc l p x')
                else:
                    F = _apply_dedup_gram(*self.gram_dedup[g],
                                          F.reshape(d - c, self.K_l, P_, nx))
                F = self._ifftc(F.reshape(d - c, self.K_l, ny, nz, nx).contiguous())
                if self.ghost_maps is None:
                    out_l += einsum(F, mps_c.conj(),
                                    'nc nsub ..., nc ... -> nsub ...')
                else:
                    out_l += einsum(F, mps_c.conj(), self.ghost_maps[g, 0].conj(),
                                    'nc nsub ..., nc ..., ... -> nsub ...')
                del F
        return self._w_adj(out_l)





# =============================================================================
# SUPERSEDED: the original subspace_cart_linop, kept for reference only.
#
# This file defined `class subspace_cart_linop` TWICE -- once here (originally
# near the top, before subspace_cart_linop_fastb0) and once again further down.
# Python binds the name to whichever definition executes last, so THIS version
# was shadowed and never actually ran: every import of subspace_cart_linop got
# the other one. That was easy to miss and repeatedly caused edits and greps to
# land on unreachable code, so it has been commented out and moved here.
#
# The live implementation is `class subspace_cart_linop` earlier in this file.
# It had already diverged substantially from this copy before any of the 2026-09
# performance work (676 vs 257 lines, ~454 lines present only in the live one),
# so this is NOT a pristine "before" snapshot of the current class -- it is an
# older lineage. Do not treat the two as interchangeable.
#
# Nothing references this code. It is retained purely so the earlier lineage is
# still readable; delete it whenever you are confident you no longer need it.
# =============================================================================
#

# class subspace_cart_linop(linop):
#
#     def __init__(self,
#                  im_size: tuple,
#                  mask_sample: torch.Tensor,
#                  mps: torch.Tensor,
#                  phi: torch.Tensor,
#                  motion_bin: torch.Tensor,
#                  dcf: Optional[torch.Tensor] = None,
#                  imperf_model: Optional[tuple] = None,
#                  imperf_model_motion: Optional[imperfection] = None,
#                  use_toeplitz: Optional[bool] = False,
#                  bparams: Optional[batching_params] = batching_params()):
#         """
#         Parameters
#         ----------
#         im_size : tuple 
#             image dims as tuple of ints (dim1, dim2, ...)
#         mask_sample : torch.tensor <float> | GPU
#             The k-space trajectory with shape (npe, ntr, nro, d). 
#                 we assume that trj values are in [-n/2, n/2] (for nxn grid)
#         mps : torch.tensor <complex> | GPU
#             sensititvity maps with shape (ncoil, ndim1, ..., ndimN)
#         phi : torch.tensor <complex> | GPU
#             subspace basis with shape (nsub, ntr)   
#         motion_bin : torch.tensor <int> | GPU
#             which pe belongs to which motion states (1,npe)
#         dcf : torch.tensor <float> | GPU
#             the density comp. functon with shape (nro, ...)
#         nufft : NUFFT
#             the nufft object, defaults to torchkbnufft
#         imperf_model : a struct cattying B0 and TEs
#
#         imperf_model_motion : motion operator
#         use_toeplitz : bool
#             toggles toeplitz normal operator
#         bparams : batching_params
#             contains the batch sizes for the coils, subspace coeffs, and field segments
#         """
#
#         ishape = (phi.shape[0], *im_size)
#         oshape = (mps.shape[0], motion_bin.shape[0],mps.shape[-1])
#         super().__init__(ishape, oshape)
#
#         # Consts
#         torch_dev = mps.device
#         assert phi.device == torch_dev
#         assert mps.device == torch_dev
#
#         # Default params
#         # if nufft is None:
#         #     nufft = torchkb_nufft(im_size, torch_dev.index)
#         if dcf is None:
#             dcf = torch.ones((motion_bin.shape[0],mps.shape[-1]), dtype=torch.float32, device=torch_dev)
#         else:
#             assert dcf.device == torch_dev
#
#         if imperf_model is None:
#             b0 = torch.zeros((1,*mps.shape), dtype=torch.float32, device=torch_dev)
#             tes = torch.zeros((phi.shape[1]), dtype=torch.float32, device=torch_dev)
#             phs0 = torch.zeros((1,*mps.shape), dtype=torch.float32, device=torch_dev)
#             phst = torch.zeros((phi.shape[1]), dtype=torch.float32, device=torch_dev)
#         else:
#             b0 = imperf_model[0]
#             tes = imperf_model[1]
#             phs0 = imperf_model[2]
#             phst = imperf_model[3]
#
#         # Rescale and type cast
#         mask_sample = mask_sample.type(torch.float32)
#         mask_sample = mask_sample>0
#         print(mask_sample.shape)
#         num_sample = motion_bin.shape[0]
#         dcf = dcf.type(torch.float32)
#         mps = mps.type(torch.complex64)
#         phi = phi.type(torch.complex64)
#         b0 = b0.type(torch.complex64)
#         tes = tes.type(torch.complex64)
#         phs0 = phs0.type(torch.complex64)
#         phst = phst.type(torch.complex64)
#         motion_bin = motion_bin.type(torch.int)
#         nt = phi.shape[1]
#         nc = mps.shape[0]
#         nx = mps.shape[-1]
#         nz = mps.shape[-2]
#         ny = mps.shape[-3]
#         nmot = b0.shape[0]
#
#         # Compute toeplitz kernels
#         self.toep_kerns = None
#
#         # Save
#         self.im_size = im_size
#         self.use_toeplitz = use_toeplitz
#         self.mask_sample = mask_sample
#         self.num_sample = num_sample
#         self.phi = phi
#         self.mps = mps
#         self.dcf = dcf
#         self.tes = tes
#         self.b0 = b0
#         self.phs0 = phs0
#         self.phst = phst
#         self.motion_bin = motion_bin
#         self.imperf_model_motion = imperf_model_motion
#         # self.nm = phs.shape[0]
#         # self.nt = phi.shape[1]
#         self.bparams = bparams
#         self.torch_dev = torch_dev
#
#     def forward(self,
#                 alphas: torch.Tensor) -> torch.Tensor:
#
#         # Useful constants
#         nt = self.phi.shape[1]
#         nc = self.mps.shape[0]
#         nx = self.mps.shape[-1]
#         nz = self.mps.shape[-2]
#         ny = self.mps.shape[-3]
#         nmot = self.b0.shape[0]
#         num_sample = self.num_sample
#         motion_bin = self.motion_bin
#         coil_batch_size = self.bparams.coil_batch_size
#         # sub_batch_size = self.bparams.sub_batch_size
#         seg_batch_size = self.bparams.field_batch_size
#         # motion_batch_size = self.bparams.motion_batch_size
#
#         # Result array
#         ksp = torch.zeros((nc, num_sample,nx), dtype=torch.complex64, device=self.torch_dev)
#
#         # Batch over coils
#         for c, d in batch_iterator(nc, coil_batch_size):
#             # mps = self.mps[c:d]
#             # Batch over time segments
#             # for a, b in batch_iterator(nmot, motion_batch_size):
#
#             for l1, l2 in batch_iterator(nt, seg_batch_size):
#                 len_t = l2-l1
#                 # FSTBPx_t = torch.zeros((nmot, d-c, len_t, ny,nz,nx), dtype=torch.complex64, device=self.torch_dev)
#                 # phi
#                 Px = einsum(alphas,self.phi[:,l1:l2], 'nsub nro npe nx, nsub nt -> nt nro npe nx')
#                 # phs 
#                 # phs = torch.exp(1j *  (2 * torch.pi * einsum(self.b0, self.tes[l1:l2],'nmot nro npe nx, nt -> nmot nt nro npe nx') + einsum(self.phs0, self.phst[l1:l2],'nmot nro npe nx, nt -> nmot nt nro npe nx')))
#                 phs1 =  2 * torch.pi * einsum(self.b0, self.tes[l1:l2],'nmot nro npe ntr, nt -> nmot nt nro npe ntr')
#                 phs2 = einsum(self.phs0, self.phst[l1:l2],'nmot nro npe ntr, nt -> nmot nt nro npe ntr')
#                 phs =  torch.exp(1j*(phs1+phs2))
#                 BPx = einsum(Px,phs, 'nt nro npe nx, nmot nt nro npe nx -> nmot nt nro npe nx')
#
#                 # motion correction
#                 if self.imperf_model_motion is None:
#                     TBPx = BPx
#                 else:
#                     TBPx = self.imperf_model_motion.apply_spatial(BPx)
#
#                 # sensitivity
#                 STBPx = einsum(TBPx, self.mps[c:d,...], 'nmot nt nro npe nx, nc nro npe nx -> nmot nc nt  nro npe nx')
#
#                 FSTBPx = fftc(STBPx, dim=(-3,-2))
#
#                 # FSTBPx_t[:,:,l1:l2,...] = FSTBPx
#                 for i in range(nmot):
#                     # Append to k-space
#                     temp = FSTBPx[i,...].reshape(d-c,len_t*ny*nz,nx)
#                     st = self.mask_sample[i,:l1,...].squeeze().sum().int()
#                     ed = self.mask_sample[i,:l2,...].squeeze().sum().int()
#                     tmp = ksp[c:d,motion_bin==i,...] 
#                     tmp[:,st:ed,...] = temp[:,self.mask_sample[i,l1:l2,...].flatten()==1,:]
#                     ksp[c:d, motion_bin==i,:] = tmp
#
#
#
#         return ksp
#
#     def adjoint(self,
#                 ksp: torch.Tensor) -> torch.Tensor:
#
#         # Useful constants
#         nt = self.phi.shape[1]
#         nsub = self.phi.shape[0]
#         nc = self.mps.shape[0]
#         nx = self.mps.shape[-1]
#         nz = self.mps.shape[-2]
#         ny = self.mps.shape[-3]
#         nmot = self.b0.shape[0]
#         motion_bin = self.motion_bin
#         coil_batch_size = self.bparams.coil_batch_size
#         # sub_batch_size = self.bparams.sub_batch_size
#         seg_batch_size = self.bparams.field_batch_size
#         # motion_batch_size = self.bparams.motion_batch_size
#         # Result subspace coefficients
#         alphas = torch.zeros((nsub, *self.im_size), dtype=torch.complex64, device=self.torch_dev)  
#         # Batch over coils
#         for c, d in batch_iterator(nc, coil_batch_size):
#             mps = self.mps[c:d]
#             # Batch over time segments
#             # SFWy = torch.zeros((nmot, nt, ny, nz,nx), dtype=torch.complex64, device=self.torch_dev)
#             # for i in range (nmot):
#             #     ksp_weighted = torch.zeros((d-c,nt*ny*nz,nx), dtype=torch.complex64, device=self.torch_dev)
#             #     ksp_weighted[:,self.mask_sample[i,:].squeeze()==1,:] = ksp[c:d,motion_bin==i,...] 
#             #     ksp_weighted = ksp_weighted.reshape(d-c,nt,ny,nz,nx)
#
#             #     for l1, l2 in batch_iterator(nt, seg_batch_size):                 
#             #         len_t = l2-l1           
#             #         FWy = ifftc(ksp_weighted[:,l1:l2,...], dim=(-3,-2)) # nc nsub nseg *im_size
#             #         # Conjugate maps
#             #         SFWy[i,l1:l2,...] = einsum(FWy, mps.conj(), 'nc nt ..., nc ... -> nt ...')
#             # if self.imperf_model_motion is None:
#             #     TSFWy = SFWy
#             # else:
#             #     TSFWy = self.imperf_model_motion.apply_spatial_adjoint(SFWy)
#             # # Conjugate imperfection maps
#             # phs = torch.exp(1j * 2 * torch.pi * einsum(self.b0, self.tes,'nmot nro npe ntr, nt -> nmot nt nro npe ntr'))
#             # BTSFWy = einsum(TSFWy,phs.conj(),'nmot nt ..., nmot nt ... -> nt ...')
#             # PBTSFWy = einsum(BTSFWy,self.phi, 'nt nro npe ntr, nsub nt -> nsub nro npe ntr')
#
#             # # Append to image
#             # alphas+= PBTSFWy
#
#             for l1, l2 in batch_iterator(nt, seg_batch_size):   
#                 len_t = l2-l1        
#                 SFWy = torch.zeros((nmot, len_t, ny, nz,nx), dtype=torch.complex64, device=self.torch_dev)
#                 for i in range (nmot):
#                     ksp_weighted = torch.zeros((d-c,len_t*ny*nz,nx), dtype=torch.complex64, device=self.torch_dev)
#
#                     st = self.mask_sample[i,:l1,...].squeeze().sum().int()
#                     ed = self.mask_sample[i,:l2,...].squeeze().sum().int()
#
#                     tmp = ksp[c:d,motion_bin==i,...] 
#
#                     ksp_weighted[:,self.mask_sample[i,l1:l2,...].flatten()==1,:] = tmp[:,st:ed,...] 
#
#                     ksp_weighted = ksp_weighted.reshape(d-c,len_t,ny,nz,nx)
#
#                     FWy = ifftc(ksp_weighted, dim=(-3,-2)) # nc nsub nseg *im_size
#                     # Conjugate maps
#                     SFWy[i,...] = einsum(FWy, mps.conj(), 'nc nt ..., nc ... -> nt ...')
#                 if self.imperf_model_motion is None:
#                     TSFWy = SFWy
#                 else:
#                     TSFWy = self.imperf_model_motion.apply_spatial_adjoint(SFWy)
#                 # Conjugate imperfection maps
#                 phs1 =  2 * torch.pi * einsum(self.b0, self.tes[l1:l2],'nmot nro npe ntr, nt -> nmot nt nro npe ntr')
#                 phs2 = einsum(self.phs0, self.phst[l1:l2],'nmot nro npe ntr, nt -> nmot nt nro npe ntr')
#                 phs =  torch.exp(1j*(phs1+phs2))
#                 BTSFWy = einsum(TSFWy,phs.conj(),'nmot nt ..., nmot nt ... -> nt ...')
#                 PBTSFWy = einsum(BTSFWy,self.phi[:,l1:l2], 'nt nro npe ntr, nsub nt -> nsub nro npe ntr')
#
#                 # Append to image
#                 alphas+= PBTSFWy
#
#         return alphas
#
#     def normal(self,
#                alphas: torch.Tensor) -> torch.Tensor:
#         return self.adjoint(self.forward(alphas))
