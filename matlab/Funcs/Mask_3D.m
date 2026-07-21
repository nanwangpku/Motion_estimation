function [ a ] = Mask_3D( a, mask_sample)
% a:           dim nreadout, Nx, Ncoils
% mask_sample: dim Ny, Nz, Nt,Nx, 1 
% [Ny,Nz,Nt,Nx,Ncoils] = size(a);
% a = reshape(a,[],Nx,Ncoils);
mask = logical(reshape(mask_sample(:,:,:,1,1,1),[],1));
a = a(mask(:,1),:,:);


end

