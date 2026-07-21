import torch
from einops import rearrange, einsum

# --- Rotation matrices ---
def rotx(t):
    ct = torch.cos(t)
    st = torch.sin(t)
    R = torch.tensor([[1,0,0],
                      [0,ct,-st],
                      [0,st,ct]], dtype=torch.float32)
    return R

def roty(t):
    ct = torch.cos(t)
    st = torch.sin(t)
    R = torch.tensor([[ct,0,st],
                      [0,1,0],
                      [-st,0,ct]], dtype=torch.float32)
    return R

def rotz(t):
    ct = torch.cos(t)
    st = torch.sin(t)
    R = torch.tensor([[ct,-st,0],
                      [st,ct,0],
                      [0,0,1]], dtype=torch.float32)
    return R

def motion_ops_approx(img_size:tuple,
               motion_params:torch.Tensor, 
               trj:torch.Tensor,
               motion_bin:torch.Tensor):
    """
    trj: torch tensor in shape nshot, nread, 3, the three trj should match with img size and motion_paramsters
    """
    
    torch_dev = motion_params.get_device()
    nbins = motion_params.shape[0]
    nshot = motion_bin.shape[0]
    nrd = trj.shape[1]
    N1,N2,N3 = img_size[:]

    trj_norm = einsum(trj.clone(),torch.tensor([1/N1,1/N2,1/N3]).to(torch_dev),"... naxis, naxis -> ... naxis")
    # x = torch.arange(-N1//2,N1//2).to(torch_dev)
    # y = torch.arange(-N2//2,N2//2).to(torch_dev)
    # z = torch.arange(-N3//2,N3//2).to(torch_dev)
    # X,Y,Z = torch.meshgrid(x,y,z,indexing='ij')
    # X = X.to(torch_dev)
    # Y = Y.to(torch_dev)
    # Z = Z.to(torch_dev)

    tmp_trans = torch.zeros((nshot, 3), device=torch_dev)
    # R = torch.zeros((nshot,nrd, 3,3), device=torch_dev)

    trj_rot = torch.zeros_like(trj)

    for ii_bin in range(nbins):
        tmpt = motion_params[ii_bin, 3:6].clone().to(torch_dev)
        tmpt = tmpt[[1,2,0]] * torch.tensor([-1,-1,1], device=torch_dev)
        # print(tmpt)
        tmp_trans[motion_bin==ii_bin,:] = tmpt

        tmpm = motion_params[ii_bin, 0:3].clone().to(torch_dev) / 180 * torch.pi
        Rx = rotx(-tmpm[2])
        Ry = roty(-tmpm[1])
        Rz = rotz(-tmpm[0])
        R = (Rz @ Ry @ Rx).t().to(torch_dev)
        trj_rot[motion_bin==ii_bin,:,:] = einsum(trj_norm[motion_bin==ii_bin,:,:], R, "... naxis, naxis nnew -> ... nnew")
        # R[ii_bin] = (Rz @ Ry @ Rx).t().to(torch_dev)[None,...].expand(nrd,3,3)
        # if ii_bin==1:
        #     print(R[ii_bin,1])
    
    # trj_rot = einsum(trj_norm, R, "nshot nx naxis, nshot nx naxis nnew -> nshot nx nnew")
    trj_rot = einsum(trj_rot, torch.tensor([N1,N2,N3]).to(torch_dev), "... naxis, naxis -> ... naxis")


    trans_phase = torch.exp(-1j*2*torch.pi*einsum(trj_norm,tmp_trans,"nshot nx naxis,nshot naxis -> nshot nx"))
    return trj_rot,trans_phase


