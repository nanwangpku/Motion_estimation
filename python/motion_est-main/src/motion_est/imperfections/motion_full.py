# TODO work in progress
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
from einops import rearrange, einsum
from motion_est.utils import (
    rotation_matrix,
    quantize_data,
    gen_grd
)
from motion_est.fourier import fft, ifft
from motion_est.imperfections.imperfection import imperfection

def affine_mats_from_parms(motion_params: torch.Tensor) -> torch.Tensor:
    """
    Returns rotation matrices from angles

    Parameters:
    -----------
    motion_params : torch.Tensor
        The motion parameters with shape (..., 6). 
        First three motion params are translations in x, y, z in range [-1, 1].
        Last three motion params are rotations in x, y, z in degrees.
    
    Returns:
    --------
    affine_mats : torch.Tensor
        The affine matrices with shape (..., 3, 4)
        where the first 3 columns is the rotation matrix, 
        and last column is the translation
    """

    # Split into translations and rotations
    batch_size = motion_params.shape[:-1]
    motion_params = motion_params.reshape(-1, 6)
    angles = motion_params[..., 3:]
    shifts = motion_params[..., :3]

    # Affine mat to return
    affine_mats = torch.zeros((motion_params.shape[0], 3, 4), 
                              device=motion_params.device, 
                              dtype=motion_params.dtype)

    # Make rotation matrices
    tup = (None,) * len(batch_size) + (slice(None),)
    ax_x = torch.tensor([1.0, 0, 0], device=motion_params.device, dtype=motion_params.dtype)[tup]
    ax_y = torch.tensor([0, 1.0, 0], device=motion_params.device, dtype=motion_params.dtype)[tup]
    ax_z = torch.tensor([0, 0, 1.0], device=motion_params.device, dtype=motion_params.dtype)[tup]
    R_x = rotation_matrix(axis=ax_x, 
                              theta=torch.deg2rad(angles[..., 0]))
    R_y = rotation_matrix(axis=ax_y, 
                            theta=torch.deg2rad(angles[..., 1]))
    R_z = rotation_matrix(axis=ax_z, 
                            theta=torch.deg2rad(angles[..., 2]))
    R = R_x @ R_y @ R_z

    # Append translations
    affine_mats[..., :3, :3] = R
    affine_mats[..., :3, -1] = shifts

    return affine_mats.reshape((*batch_size, 3, 4))

def apply_affine_3d(imgs: torch.Tensor,
                    affine_mats: torch.Tensor) -> torch.Tensor:
    """
    Applies affine transformations to 3D volumes

    Parameters:
    -----------
    imgs : torch.Tensor
        The volumes with shape (N, *im_batch, nx, ny, nz)
        where nx ny nz are the spatial image dims
    affine_mats : torch.Tensor
        The affine matrices with shape (N, *affine_batch, 3, 4)
        where the first 3 columns is the rotation matrix, 
        and last column is the translation
    
    Returns:
    --------
    tformed_imgs : torch.Tensor
        The transformed volumes with shape (N, *im_batch, *affine_batch, *im_size)
    """

    # Remember original shapes
    im_batch = imgs.shape[1:-3]
    affine_batch = affine_mats.shape[1:-2]
    N = imgs.shape[0]
    assert imgs.shape[0] == affine_mats.shape[0]

    # Flip image from x y z to z y x
    imgs = rearrange(imgs, '... nx ny nz -> ... nz ny nx')
    im_size = imgs.shape[-3:]

    # Flatten
    imgs = imgs.reshape(N, -1, *im_size)
    affine_mats = affine_mats.reshape(N, -1, 3, 4)
    A = affine_mats.shape[1]
    C = imgs.shape[1]

    # Apply affine transformations
    align_corners = False
    grid = F.affine_grid(affine_mats.reshape(-1, 3, 4), 
                         (N * A, imgs.shape[1], *im_size), align_corners=align_corners)
    grid = rearrange(grid, '(N A) nz ny nx three -> N (A nz) ny nx three',
                     N=N, A=A)
    tformed_imgs_real = F.grid_sample(imgs.real, grid, align_corners=align_corners)
    tformed_imgs_imag = F.grid_sample(imgs.imag, grid, align_corners=align_corners)
    tformed_imgs = tformed_imgs_real + 1j * tformed_imgs_imag

    # Reshape batch dims
    tformed_imgs = rearrange(tformed_imgs, 'N C (A nz) ny nx -> A C nz ny nx N',
                             A=A, nz=im_size[0])
    tformed_imgs = tformed_imgs.reshape((*affine_batch, *tformed_imgs.shape[1:]))
    tformed_imgs = rearrange(tformed_imgs, '... C nz ny nx N -> C ... nz ny nx N')
    tformed_imgs = tformed_imgs.reshape((*im_batch, *tformed_imgs.shape[1:]))
    tformed_imgs = rearrange(tformed_imgs, '... nz ny nx N -> N ... nz ny nx')

    # Flip back to x y z
    tformed_imgs = rearrange(tformed_imgs, '... nz ny nx -> ... nx ny nz')

    return tformed_imgs


def fftc(x, dim):
    return torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(x, dim=dim), dim=dim), dim=dim)

def ifftc(x, dim):
    return torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(x, dim=dim), dim=dim), dim=dim)

def motion_ops(img_size:tuple,
               motion_params:torch.Tensor, 
               pad_num:tuple):
    

    img_size0 = img_size
    img_size = img_size0 + pad_num * 2 
    torch_dev = motion_params.get_device()


    N1,N2,N3 = img_size[:]

    x = torch.arange(-N1//2,N1//2).to(torch_dev)
    y = torch.arange(-N2//2,N2//2).to(torch_dev)
    z = torch.arange(-N3//2,N3//2).to(torch_dev)
    X,Y,Z = torch.meshgrid(x,y,z,indexing='ij')
    X = X.to(torch_dev)
    Y = Y.to(torch_dev)
    Z = Z.to(torch_dev)

    kGrid = (X*2*torch.pi/N1, Y*2*torch.pi/N2, Z*2*torch.pi/N3)
    rGrid = (X,Y,Z)
    rkGrid = [[None for _ in range(3)] for _ in range(2)]

    per = [[0, 2, 1], [1, 0, 2]]
    for n in range(2):
        for m in range(3):
            rkGrid[n][m] = rGrid[per[1 - n][m] ] * kGrid[per[n][m]]
    
    if not isinstance(motion_params, torch.Tensor):
        pos_rot = torch.tensor(motion_params[:3])* torch.pi / 180
        pos_tra = torch.tensor(motion_params[[4,5,3]])
    else:
        pos_rot = motion_params[:3] * torch.pi / 180
        pos_tra = motion_params[[4,5,3]]
 
    pos_tra = pos_tra * torch.tensor([-1, -1, 1]).to(torch_dev)
    pos_tra = pos_tra * torch.tensor(img_size) / torch.tensor(img_size0)
    
    et = [torch.exp(-1j * (kGrid[0] * pos_tra[0] + kGrid[1] * pos_tra[1] + kGrid[2] * pos_tra[2]))]
    eth = [torch.exp(1j * (kGrid[0] * pos_tra[0] + kGrid[1] * pos_tra[1] + kGrid[2] * pos_tra[2]))]

    

    theta = torch.remainder(pos_rot, 2 * torch.pi)
    tantheta2 = torch.tan(theta / 2)
    sintheta = torch.sin(theta)

    et.append([torch.exp(1j * tantheta2[m] * rkGrid[0][m]) for m in range(3)])
    et.append([torch.exp(-1j * sintheta[m] * rkGrid[1][m]) for m in range(3)])

    

    def T(x):
        dims = len(x.shape)-len(img_size)
        if sum(pad_num) > 0:
            def pad_for(x):
                return torch.nn.functional.pad(x, (pad_num[2],pad_num[2],pad_num[1],pad_num[1],pad_num[0],pad_num[0]) )

            def pad_adj(x:torch.Tensor):
                # if dims>0:
                x = x[...,pad_num[0]:(pad_num[0]+img_size0[0]), pad_num[1]:(pad_num[1]+img_size0[1]), pad_num[2]:(pad_num[2]+img_size0[2])]
                # else:
                #     x = x[pad_num[0]:(pad_num[0]+img_size0[0]), pad_num[1]:(pad_num[1]+img_size0[1]), pad_num[2]:(pad_num[2]+img_size0[2])]
                return x

        else:
            pad_for = lambda x: x
            pad_adj = lambda x: x

        fft_dim = (0+dims,1+dims,2+dims)
        x = fftc(x, dim=fft_dim)
        x = pad_for(x)
        x = ifftc(x, dim=fft_dim)
        for m in range(3):
            x = fftc(x, per[0][m]+dims )
            x = x * et[1][m]
            x = ifftc(x, per[0][m]+dims )
            x = fftc(x, per[1][m]+dims )
            x = x * et[2][m]
            x = ifftc(x, per[1][m]+dims )
            x = fftc(x, per[0][m]+dims )
            x = x * et[1][m]
            x = ifftc(x, per[0][m]+dims )
        x = fftc(x, dim=fft_dim)
        x = x * et[0]
        x = pad_adj(x)
        x = ifftc(x, dim=fft_dim)
        return x

    def Th(x):
        dims = len(x.shape)-len(img_size)
        if sum(pad_num) > 0:
            def pad_for(x):
                return torch.nn.functional.pad(x, (pad_num[2],pad_num[2],pad_num[1],pad_num[1],pad_num[0],pad_num[0]) )

            def pad_adj(x:torch.Tensor):
                # if dims>0:
                x = x[...,pad_num[0]:(pad_num[0]+img_size0[0]), pad_num[1]:(pad_num[1]+img_size0[1]), pad_num[2]:(pad_num[2]+img_size0[2])]
                # else:
                #     x = x[pad_num[0]:(pad_num[0]+img_size0[0]), pad_num[1]:(pad_num[1]+img_size0[1]), pad_num[2]:(pad_num[2]+img_size0[2])]
                return x

        else:
            pad_for = lambda x: x
            pad_adj = lambda x: x

        fft_dim = (0+dims,1+dims,2+dims)
        x = fftc(x, dim=fft_dim)
        x = pad_for(x)
        x = x * torch.conj(et[0])
        x = ifftc(x, dim=fft_dim)
        for m in reversed(range(3)):
            x = fftc(x, per[0][m] +dims)
            x = x * torch.conj(et[1][m])
            x = ifftc(x, per[0][m]+dims)
            x = fftc(x, per[1][m]+dims )
            x = x * torch.conj(et[2][m])
            x = ifftc(x, per[1][m]+dims )
            x = fftc(x, per[0][m]+dims )
            x = x * torch.conj(et[1][m])
            x = ifftc(x, per[0][m]+dims )
        x = fftc(x, dim=fft_dim)
        x = pad_adj(x)
        x = ifftc(x, dim=fft_dim)
        return x
    return T, Th

class motion_full_impefection(imperfection):


    def __init__(self,
                 motion_params: torch.Tensor,
                 img_size: tuple,
                 pad_num: tuple):
        

        self.motion_params = motion_params
        self.img_size = img_size
        self.pad_num = pad_num
        self.motion_states = motion_params.shape[0]
        self.torch_dev = motion_params.device
    
    
    # def apply_spatial(self, 
    #                   x: Optional[torch.Tensor] = None) -> torch.Tensor:
    #     y = torch.zeros_like(x)
    #     for i in range(self.motion_states):
    #         T,_ = motion_ops(self.img_size, self.motion_params[i,...].squeeze(0), self.pad_num)
    #         y[i,...] = T(x[i,...])
    #     # x -> ... nc *im_size
    #     return y # -> ... nc len(ls) *im_size
    
    def apply_spatial(self, 
                      x: torch.Tensor,
                      motion_bin:Optional[tuple] = None) -> torch.Tensor:
        if motion_bin is None:
            motion_bin = tuple(range(0,self.motion_states,1))
        if len(motion_bin)>self.motion_states or max(motion_bin)>=self.motion_states:
            print('forward error')
        y = torch.zeros_like(x)
        for i in range(len(motion_bin)):
            T,_ = motion_ops(self.img_size, self.motion_params[motion_bin[i],...].squeeze(0), self.pad_num)
            y[i,...] = T(x[i,...])
        # x -> ... nc *im_size
        return y # -> ... nc len(ls) *im_size
    
    # def apply_spatial_adjoint(self, 
    #                           y: torch.Tensor) -> torch.Tensor:
    #     # y -> ... nc len(ls) *im_size
    #     x = torch.zeros_like(y)
    #     for i in range(self.motion_states):
    #         _,Th = motion_ops(self.img_size, self.motion_params[i,...], self.pad_num)
    #         x[i,...] = Th(y[i,...])

    def apply_spatial_adjoint(self, 
                              y: torch.Tensor,
                              motion_bin:Optional[tuple] = None) -> torch.Tensor:
        # y -> ... nc len(ls) *im_size
        if motion_bin is None:
            motion_bin = tuple(range(0,self.motion_states,1))

        if len(motion_bin)>self.motion_states or max(motion_bin)>=self.motion_states:
            print('adjoint error')
        x = torch.zeros_like(y)
        for i in range(len(motion_bin)):
            _,Th = motion_ops(self.img_size, self.motion_params[motion_bin[i],...], self.pad_num)
            x[i,...] = Th(y[i,...])
        return x
    


class motion_app_impefection(imperfection):

# this is to apply imperfections for approximate motion correction

    def __init__(self,
                 motion_params: torch.Tensor,
                 img_size: tuple,
                 trans_phase: torch.Tensor):
        

        self.motion_params = motion_params
        self.img_size = img_size
        self.trans_phase = trans_phase
        self.motion_states = motion_params.shape[0]
        self.torch_dev = motion_params.device
    
    
    
    def apply_rot(self, 
                      x: torch.Tensor,
                      motion_bin:Optional[tuple] = None) -> torch.Tensor:
        # to be done
        if motion_bin is None:
            motion_bin = tuple(range(0,self.motion_states,1))
        if len(motion_bin)>self.motion_states or max(motion_bin)>=self.motion_states:
            print('forward error')
        y = torch.zeros_like(x)
        for i in range(len(motion_bin)):
            T,_ = motion_ops(self.img_size, self.motion_params[motion_bin[i],...].squeeze(0), self.pad_num)
            y[i,...] = T(x[i,...])
        # x -> ... nc *im_size
        return y # -> ... nc len(ls) *im_size
    
    def apply_trans_phase(self, 
                      x: torch.Tensor) -> torch.Tensor:
        y = x * self.trans_phase
        # x -> ... nc *im_size
        return y # -> ... nc len(ls) *im_size
    
    

    def apply_spatial_adjoint(self, 
                              y: torch.Tensor,
                              motion_bin:Optional[tuple] = None) -> torch.Tensor:
        # to be done
        # y -> ... nc len(ls) *im_size
        if motion_bin is None:
            motion_bin = tuple(range(0,self.motion_states,1))

        if len(motion_bin)>self.motion_states or max(motion_bin)>=self.motion_states:
            print('adjoint error')
        x = torch.zeros_like(y)
        for i in range(len(motion_bin)):
            _,Th = motion_ops(self.img_size, self.motion_params[motion_bin[i],...], self.pad_num)
            x[i,...] = Th(y[i,...])
        return x
    
    def apply_trans_phase_adjoint(self, 
                      x: torch.Tensor) -> torch.Tensor:
        # y = x * self.trans_phase.conj()
        y = x * self.trans_phase.conj()
        # x -> ... nc *im_size
        return y # -> ... nc len(ls) *im_size

