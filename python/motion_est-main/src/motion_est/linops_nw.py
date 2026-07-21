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
    x_dim = x.shape
    return torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(x, dim=dim), dim=dim)/np.sqrt(product(x_dim,dim)), dim=dim)

def ifftc(x, dim):
    x_dim = x.shape
    return torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(x, dim=dim), dim=dim)*np.sqrt(product(x_dim,dim)), dim=dim)

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
        print(mask_sample.shape)
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
                phs1 =  2 * torch.pi * einsum(self.b0, self.tes[l1:l2],'nmot nro npe ntr, nt -> nmot nt nro npe ntr')
                phs2 = einsum(self.phs0, self.phst[l1:l2],'nmot nro npe ntr, nt -> nmot nt nro npe ntr')
                phs =  torch.exp(1j*(phs1+phs2))
                BPx = einsum(Px,phs, 'nt nro npe nx, nmot nt nro npe nx -> nmot nt nro npe nx')

                # motion correction
                if self.imperf_model_motion is None:
                    TBPx = BPx
                else:
                    TBPx = self.imperf_model_motion.apply_spatial(BPx)

                # sensitivity
                STBPx = einsum(TBPx, self.mps[c:d,...], 'nmot nt nro npe nx, nc nro npe nx -> nmot nc nt  nro npe nx')

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
            # Batch over time segments
            # SFWy = torch.zeros((nmot, nt, ny, nz,nx), dtype=torch.complex64, device=self.torch_dev)
            # for i in range (nmot):
            #     ksp_weighted = torch.zeros((d-c,nt*ny*nz,nx), dtype=torch.complex64, device=self.torch_dev)
            #     ksp_weighted[:,self.mask_sample[i,:].squeeze()==1,:] = ksp[c:d,motion_bin==i,...] 
            #     ksp_weighted = ksp_weighted.reshape(d-c,nt,ny,nz,nx)
                
            #     for l1, l2 in batch_iterator(nt, seg_batch_size):                 
            #         len_t = l2-l1           
            #         FWy = ifftc(ksp_weighted[:,l1:l2,...], dim=(-3,-2)) # nc nsub nseg *im_size
            #         # Conjugate maps
            #         SFWy[i,l1:l2,...] = einsum(FWy, mps.conj(), 'nc nt ..., nc ... -> nt ...')
            # if self.imperf_model_motion is None:
            #     TSFWy = SFWy
            # else:
            #     TSFWy = self.imperf_model_motion.apply_spatial_adjoint(SFWy)
            # # Conjugate imperfection maps
            # phs = torch.exp(1j * 2 * torch.pi * einsum(self.b0, self.tes,'nmot nro npe ntr, nt -> nmot nt nro npe ntr'))
            # BTSFWy = einsum(TSFWy,phs.conj(),'nmot nt ..., nmot nt ... -> nt ...')
            # PBTSFWy = einsum(BTSFWy,self.phi, 'nt nro npe ntr, nsub nt -> nsub nro npe ntr')
            
            # # Append to image
            # alphas+= PBTSFWy

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
                if self.imperf_model_motion is None:
                    TSFWy = SFWy
                else:
                    TSFWy = self.imperf_model_motion.apply_spatial_adjoint(SFWy)
                # Conjugate imperfection maps
                phs1 =  2 * torch.pi * einsum(self.b0, self.tes[l1:l2],'nmot nro npe ntr, nt -> nmot nt nro npe ntr')
                phs2 = einsum(self.phs0, self.phst[l1:l2],'nmot nro npe ntr, nt -> nmot nt nro npe ntr')
                phs =  torch.exp(1j*(phs1+phs2))
                BTSFWy = einsum(TSFWy,phs.conj(),'nmot nt ..., nmot nt ... -> nt ...')
                PBTSFWy = einsum(BTSFWy,self.phi[:,l1:l2], 'nt nro npe ntr, nsub nt -> nsub nro npe ntr')
            
                # Append to image
                alphas+= PBTSFWy

        return alphas
    
    def normal(self,
               alphas: torch.Tensor) -> torch.Tensor:
        return self.adjoint(self.forward(alphas))
    

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
        print(mask_sample.shape)
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
                phs1 =  2 * torch.pi * einsum(self.b0, self.tes[l1:l2],'nmot nro npe ntr, nt -> nmot nt nro npe ntr')
                phs2 = einsum(self.phs0, self.phst[l1:l2],'nmot nro npe ntr, nt -> nmot nt nro npe ntr')
                phs =  torch.exp(1j*(phs1+phs2))
                BPx = einsum(Px,phs, 'nt nro npe nx, nmot nt nro npe nx -> nmot nt nro npe nx')

                # motion correction
                if self.imperf_model_motion is None:
                    TBPx = BPx
                else:
                    TBPx = self.imperf_model_motion.apply_spatial(BPx)

                # sensitivity
                STBPx = einsum(TBPx, self.mps[c:d,...], 'nmot nt nro npe nx, nc nro npe nx -> nmot nc nt  nro npe nx')

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
            # Batch over time segments
            # SFWy = torch.zeros((nmot, nt, ny, nz,nx), dtype=torch.complex64, device=self.torch_dev)
            # for i in range (nmot):
            #     ksp_weighted = torch.zeros((d-c,nt*ny*nz,nx), dtype=torch.complex64, device=self.torch_dev)
            #     ksp_weighted[:,self.mask_sample[i,:].squeeze()==1,:] = ksp[c:d,motion_bin==i,...] 
            #     ksp_weighted = ksp_weighted.reshape(d-c,nt,ny,nz,nx)
                
            #     for l1, l2 in batch_iterator(nt, seg_batch_size):                 
            #         len_t = l2-l1           
            #         FWy = ifftc(ksp_weighted[:,l1:l2,...], dim=(-3,-2)) # nc nsub nseg *im_size
            #         # Conjugate maps
            #         SFWy[i,l1:l2,...] = einsum(FWy, mps.conj(), 'nc nt ..., nc ... -> nt ...')
            # if self.imperf_model_motion is None:
            #     TSFWy = SFWy
            # else:
            #     TSFWy = self.imperf_model_motion.apply_spatial_adjoint(SFWy)
            # # Conjugate imperfection maps
            # phs = torch.exp(1j * 2 * torch.pi * einsum(self.b0, self.tes,'nmot nro npe ntr, nt -> nmot nt nro npe ntr'))
            # BTSFWy = einsum(TSFWy,phs.conj(),'nmot nt ..., nmot nt ... -> nt ...')
            # PBTSFWy = einsum(BTSFWy,self.phi, 'nt nro npe ntr, nsub nt -> nsub nro npe ntr')
            
            # # Append to image
            # alphas+= PBTSFWy

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
                if self.imperf_model_motion is None:
                    TSFWy = SFWy
                else:
                    TSFWy = self.imperf_model_motion.apply_spatial_adjoint(SFWy)
                # Conjugate imperfection maps
                phs1 =  2 * torch.pi * einsum(self.b0, self.tes[l1:l2],'nmot nro npe ntr, nt -> nmot nt nro npe ntr')
                phs2 = einsum(self.phs0, self.phst[l1:l2],'nmot nro npe ntr, nt -> nmot nt nro npe ntr')
                phs =  torch.exp(1j*(phs1+phs2))
                BTSFWy = einsum(TSFWy,phs.conj(),'nmot nt ..., nmot nt ... -> nt ...')
                PBTSFWy = einsum(BTSFWy,self.phi[:,l1:l2], 'nt nro npe ntr, nsub nt -> nsub nro npe ntr')
            
                # Append to image
                alphas+= PBTSFWy

        return alphas
    
    def normal(self,
               alphas: torch.Tensor) -> torch.Tensor:
        return self.adjoint(self.forward(alphas))
    

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
        print(mask_sample.shape)
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