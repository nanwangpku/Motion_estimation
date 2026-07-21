from typing import Union, Optional

import numpy as np
import torch
import os

def readcfl(name):
    # get dims from .hdr
    h = open(name + ".hdr", "r")
    h.readline() # skip
    l = h.readline()
    h.close()
    dims = [int(i) for i in l.split( )]

    # remove singleton dimensions from the end
    n = np.prod(dims)
    dims_prod = np.cumprod(dims)
    dims = dims[:np.searchsorted(dims_prod, n)+1]

    # load data and reshape into dims
    d = open(name + ".cfl", "r")
    a = np.fromfile(d, dtype=np.complex64, count=n);
    d.close()
    return a.reshape(dims, order='F') # column-major

	
def writecfl(name, array):
    h = open(name + ".hdr", "w")
    h.write('# Dimensions\n')
    for i in (array.shape):
            h.write("%d " % i)
    h.write('\n')
    h.close()
    d = open(name + ".cfl", "w")
    array.T.astype(np.complex64).tofile(d) # tranpose for column-major order
    d.close()



def tonp(x: Union[np.ndarray, torch.tensor]):
    """
    Convert to numpy array
    """
    if torch.is_tensor(x):
        # resolve conj if needed
        if x.is_complex():
            x = x.detach().cpu().resolve_conj().numpy()
        else:
            x = x.detach().cpu().numpy()
        return x
    elif isinstance(x, (list, tuple)):
        return np.array(x)
    else:
        return x

def totorch(x: Union[np.ndarray, torch.tensor], device, dtype=None, precision='float32'):
    """
    Convert to torch on device with specific datatype
    """
    assert precision in ['float16', 'float32', 'float64']
    if precision == 'float16':
        default_complex = torch.complex32
        default_float = torch.float16
    elif precision == 'float32':
        default_complex = torch.complex64
        default_float = torch.float32
    else:
        default_complex = torch.complex128
        default_float = torch.float64

    if dtype is not None:
        print("WARNING in totorch(): dtype is depreciated. Use precision instead.")

    if torch.is_tensor(x):
        x = x.to(device)
    elif isinstance(x, (list, tuple)):
        x = torch.from_numpy(np.array(x)).to(device)
    else:
        x = torch.from_numpy(x).to(device)

    if dtype is not None:
        x = x.to(dtype)
    else:
        if torch.is_complex(x):
            x = x.to(default_complex)
        else:
            x = x.to(default_float)

    return x

def get_totorch(device, dtype=None, precision='float32'):
    """
    Get callable totorch function for a specific device and datatype
    """
    def totorch_func(x):
        return totorch(x, device, dtype=dtype, precision=precision)
    
    return totorch_func

def load_to_backend(x: Union[np.ndarray, torch.tensor], backend='np', allow_return_complex=True):
    """
    Load data to desired backend

    Parameters
    ----------
    x : Union[np.ndarray, torch.tensor]
        Data to be loaded
    backend : str, optional
        Desired backend, by default 'np'. Can be 'np' or 'torch'

    Returns
    -------
    y : Union[np.ndarray, torch.tensor]
        Data loaded to desired backend
    ret_info : tuple
        Information about input for returning to original backend: (originalBackend, inputDtype, inputDevice)
    """

    assert backend in ['np', 'torch'], "Backend must be 'np' or 'torch'"

    ret_dtype = x.dtype

    # for when the process converts complex input to real-valued output
    if (((torch.is_tensor(x) and torch.is_complex(x)) or (isinstance(x, np.ndarray) and np.iscomplexobj(x))) and not allow_return_complex):
        ret_dtype = x.real.dtype

    if backend == 'np':
        """
        Convert to np array
        """
        if torch.is_tensor(x):
            y = tonp(x)
            ret_info = ('torch', ret_dtype, x.device)
        else:
            y = x
            ret_info = ('np', ret_dtype, None)

    elif backend == 'torch':
        """
        Convert to torch tensor
        """
        if torch.is_tensor(x):
            y = x
            ret_info = ('torch', ret_dtype, x.device)
        else:
            y = torch.from_numpy(x)
            ret_info = ('np', ret_dtype, 'cpu')
    
    return y, ret_info

def return_to_backend(y, ret_info, keep_complex=True):
    """
    Return data to original backend

    Parameters
    ----------
    y : Union[np.ndarray, torch.tensor]
        Data to be returned to original backend
    ret_info : tuple
        Information about input for returning to original backend: (originalBackend, inputDtype, inputDevice)
    keep_complex : bool, optional
        If y is complex but ret_info[1] is real, return y as complex, by default True
    
    Returns
    -------
    x : Union[np.ndarray, torch.tensor]
        Data returned to original backend
    """
    # return to np
    if ret_info[0] == 'np':
        if torch.is_tensor(y):
            y = tonp(y)

        if not (np.iscomplexobj(y) and keep_complex):
            y = y.astype(ret_info[1])
        
        return y

    # return to torch
    else:
        if torch.is_tensor(y):
            y = y.to(ret_info[2])
        else:
            y = torch.from_numpy(y).to(ret_info[2])
        
        if not (torch.is_complex(y) and keep_complex):
            y = y.type(ret_info[1])
        
        return y

def batch_iterator(total: int, 
                   batch_size: int):
    """
    Get iteratable list of indices for batched iteration

    Parameters
    ----------
    total : int
        Total number of elements to iterate over
    batch_size : int
        Batch size
    
    Returns
    -------
    batch_list : list[tuple]
        List of tuples of the form (start, end) where start is the
        starting index of the batch and end is the ending index of the batch.
    """
    assert total > 0, f'batch_iterator called with {total} elements'
    delim = list(range(0, total, batch_size)) + [total]
    return zip(delim[:-1], delim[1:])

def torch_resize(input: torch.tensor, 
                oshape: tuple):
    """Resize with zero-padding or cropping.
    Repurposed from sigpy 

    Args:
        input (torch.tensor): Input array.
        oshape (tuple of ints): Output shape.

    Returns:
        torch.tensor: Zero-padded or cropped result.
    """

    input, ret_info = load_to_backend(input, 'torch')

    assert len(input.shape) == len(oshape), \
        "Input and output must have same number of dimensions."
    
    ishape = input.shape

    if ishape == oshape: return input

    ishift = [max(i // 2 - o // 2, 0) for i, o in zip(ishape, oshape)]
    oshift = [max(o // 2 - i // 2, 0) for i, o in zip(ishape, oshape)]
    
    copy_shape = [min(i - si, o - so)
                  for i, si, o, so in zip(ishape, ishift, oshape, oshift)]
        
    islice = tuple([slice(si, si + c) for si, c in zip(ishift, copy_shape)])
    oslice = tuple([slice(so, so + c) for so, c in zip(oshift, copy_shape)])

    output = torch.zeros(oshape, dtype=input.dtype)
    output[oslice] = input[islice]

    return return_to_backend(output, ret_info)
 
def gen_grid(im_shape: tuple,
             fovs: Optional[Union[tuple, float]] = None):
    """
    Generate a grid of points in shape (im_shape[0], im_shape[1], ..., im_shape[d], d)
    By default, ranges from -0.5 to 0.5 in each dimension. 
    If provided, ranges from -fov/2 to fov/2 in each dimension.
    """
    if fovs is None:
        fovs = [1] * len(im_shape)
    elif isinstance(fovs, float) or isinstance(fovs, int):
        fovs = [fovs] * len(im_shape)

    lins = [fovs[i] * torch.arange(-im_shape[i]//2, im_shape[i]//2) / im_shape[i] for i in range(len(im_shape))]
    grid = torch.stack(torch.meshgrid(*lins, indexing='ij'), -1)

    return grid

def polynomial_fit(data: torch.tensor, 
                   order: int, 
                   mask: torch.tensor = None,
                   svd_thresh: float = None,
                   return_full: bool = False,
                   add_cross_terms: bool = True) -> torch.tensor:
    """
    Spatially fit (image) data to a polynomial of a given order.

    Parameters
    ----------
    data : torch.tensor
        Image data to be fit, either 2D or 3D
    order : int
        Order of polynomial to fit
    mask : torch.tensor, optional
        Mask to apply to data for fit point selection, by default None.
    svd_thresh : float, optional
        Threshold for SVD on polynomial term matrix, by default None
    return_full : bool, optional
        If true and a mask if provided, return the fit defined at every 
        image point, otherwise by default only return the fit at the mask points.
    add_cross_terms : bool, optional
        If true, include cross terms in the polynomial fit, by default True
    
    Returns
    -------
    torch.tensor
        Polynomial fit on the data of same shape as input data
    """

    # remove singleton dimensions
    full_im_shape = data.shape

    data[torch.isnan(data)] = 0
    data[torch.isinf(data)] = 0

    if mask is None:
        mask = torch.ones_like(data, dtype=bool)

    data = data.squeeze()
    mask = mask.squeeze()

    im_shape = data.shape
    d = len(im_shape)

    assert d in [2, 3], "Only 2D and 3D image data supported"
    assert order >= 0, "Order must be non-negative"

    # data
    b = data[mask]

    # build polynomial terms
    def build_poly(order, im_shape, mask=None, add_cross_terms=add_cross_terms):
        grid = gen_grid(im_shape, fovs=1).moveaxis(-1, 0).to(data.device)
        poly_terms = []
        poly_terms.append(torch.ones(im_shape).to(data.device)[mask])
        for i in range(1, order+1):
            for j in range(d):
                poly_terms.append(grid[j][mask] ** i)
                if add_cross_terms:
                    cross_term_inds = [k for k in range(d) if k != j]
                    for k in cross_term_inds:
                        poly_terms.append(grid[j][mask] ** i * grid[k][mask] ** i)

        poly_terms = torch.stack(poly_terms, dim=-1) # ... npoly
        return poly_terms
            
    # fit
    A = build_poly(order, im_shape, mask=mask).to(data.dtype).to(data.device)

    # remove null-space
    U, S, VH = torch.linalg.svd(A, full_matrices=False)
    
    if svd_thresh is not None:
        svd_sum = torch.cumsum(S, dim=0) / torch.sum(S)
        svd_thresh_idx = max(torch.where(svd_sum >= svd_thresh)[0][0].item(), 1)
        U = U[:, :svd_thresh_idx]
        S = S[:svd_thresh_idx]
        VH = VH[:svd_thresh_idx]

    # fit
    x_tild = torch.linalg.lstsq(U, b).solution

    if return_full:
        A_full = build_poly(order, im_shape, mask=None)
        A_full = A_full.to(data.dtype).to(data.device) # im_size x npoly
        U_full = A_full @ (VH.T.conj() * (S ** -1))
        out = U_full @ x_tild
    else:
        out = torch.zeros_like(data)
        out[mask] = U @ x_tild

    return out.reshape(full_im_shape)

# NW, 250406, temporary add
def dic_fit(im_orig: torch.Tensor,
            dic: torch.Tensor,
            param_inp: Optional[tuple] = None,
            ) -> torch.Tensor:

    # Move tensors to GPU if needed
    torch_dev = im_orig.device
    dic = dic.to(torch_dev)

    im_size = torch.tensor(im_orig.shape)  # Three spatial dimensions + 1 time dimension
    dic_size = torch.tensor(dic.shape)


    Nt = im_size[0]  # Time points

    down_rate = 1  # Set this accordingly
    down_select = torch.prod(im_size[1:]) // down_rate
    
    # Reshape tensors
    im_orig_2d = im_orig.reshape(Nt, torch.prod(im_size[1:]))
    norm_cal = torch.norm(im_orig_2d,2,dim=1)

    im_norm = im_orig_2d/norm_cal[:,None]# norm over second dim

    dic_2d = dic.reshape(Nt,torch.prod(dic_size[1:]))

    norm_cal = torch.norm(dic_2d,2,dim=1) # norm over second dim

    dic_norm = dic_2d/norm_cal[:,None]
    dic_norm = dic_norm.T  

    # Initialize dic_find with GPU support if required

    dic_find = torch.zeros(torch.prod(im_size[1:]), dtype=torch.int32, device=torch_dev)
    diff = torch.zeros_like(dic_find, dtype=torch.float32)

    param_num = len(param_inp)
    param_out = torch.zeros(param_num,torch.prod(im_size[1:])).to(torch_dev)

    # Iterate through downsampling steps
    for l1, l2 in batch_iterator(torch.prod(im_size[1:]) , down_select):

        # Move I_test_2d_temp to GPU if needed
        im_norm_tmp = im_norm[:,l1:l2]

        # Compute dictionary matching
        dic_test = torch.matmul(dic_norm, im_norm_tmp)

        # Find max value and its index
        temp_v, temp_loc = torch.max(torch.abs(dic_test), dim=0)

        dic_find[l1:l2] = temp_loc
        # print(f'size of dic find {dic_find.shape}')
        diff[l1:l2] = temp_v

        
        # Calculate B0s_find and OR_find
    for i in range(param_num):
        param_vec = param_inp[i].to(torch_dev)

        param_out[i,:] = param_vec[dic_find]

    # Extract matched dictionary values
    dic_p_2d = dic_2d[:,dic_find]

    # Calculate pd (scaling factor)
    p = (torch.sum(torch.abs(im_norm) ** 2, dim=0) ** 0.5) / (
        torch.sum(torch.abs(dic_p_2d) ** 2, dim=0) ** 0.5
    )
    im_size_reshape = im_size[1:]
    im_size_reshape = im_size_reshape.tolist()
    # p = p.view(im_size[1], im_size[2], im_size[3])
    # param_out = param_out.view(im_size[1], im_size[2], im_size[3])
    # diff = diff.view(im_size[1], im_size[2], im_size[3])
    p = p.view(*im_size_reshape)
    param_out = param_out.view(*im_size_reshape)
    diff = diff.view(*im_size_reshape)

    return param_out,p,diff

def dic_gen_B0(b0s: torch.Tensor,
               tes: torch.Tensor,
            ) -> torch.Tensor:

    rfp = 1  # B1 effect
    OR=0     # off-resonance effect. set  as 0
    
    # TEs = TEs-TEs(1);  # alwys start with no zero phase
    dic = torch.zeros((tes.shape[0],b0s.shape[0]), dtype=torch.complex64)
    for i_b0 in range(torch.numel(b0s)):
        fb = b0s[i_b0]
        dic[:,i_b0] = torch.exp(1j * 2 * torch.pi * fb * tes)

    return  dic


def polyfit_NthOrder(im_orig: torch.Tensor,
                     mask:  torch.Tensor,
                     order: int,) -> torch.Tensor:
    
    torch_dev = im_orig.device
    ny,nz = torch.tensor(im_orig.shape)
    mask = mask.float()
    y = torch.arange(ny).float()
    z = torch.arange(nz).float()

    Y, Z = torch.meshgrid(y-ny//2, z-nz//2, indexing='ij')  # Meshgrid for pixel coordinates
    # print(Y.shape)
    Y = Y.to(torch_dev)*mask
    Z = Z.to(torch_dev)*mask

    Y_flat,Z_flat = Y.flatten(), Z.flatten()
    img_flat = im_orig.flatten()
    mask_in = mask.flatten()
    Y_in = Y_flat[mask_in==1]
    Z_in = Z_flat[mask_in==1]
    img_in = img_flat[mask_in==1]
    # img_in = img_in
    # generate basis
    A = torch.ones_like(Y_in)[:,None]
    for curr_order in range(1,order+1,1):
        for xpower in range(curr_order+1):
            A = torch.cat((A, (Y_in**xpower * Z_in**(curr_order-xpower))[:,None] ),dim=1)

    # A = torch.stack([Y_in**2, Z_in**2, Y_in*Z_in, Y_in, Z_in, torch.ones_like(Y_in)], dim=1)
    
    results= torch.linalg.lstsq(A, img_in.unsqueeze(1))
    coeffs = results.solution

    b0_out = torch.matmul(A,coeffs)
    img_out = torch.zeros_like(img_flat)
    img_out[mask_in==1] = b0_out.squeeze().float()
    img_out = img_out.view(ny,nz)
    # print(img_out.shape)

    return img_out
