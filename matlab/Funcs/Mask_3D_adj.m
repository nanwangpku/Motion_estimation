function [ y ] = Mask_3D_adj( y, mask_sample )
% y:           dim nreadout, Nx, Ncoils
% mask_sample: dim Ny, Nz, Nt,Nx, 1 
% gpu_flag = isgpuarray(y);
% y = gather(y);
[~,~,Ncoils] = size(y);
[Ny,Nz,Nt,Nx,~] = size(mask_sample);
mask = gather(logical(reshape(mask_sample(:,:,:,1,1,1),[],1)));
% mask = logical(reshape(mask_sample(:,:,:,1,1,1),[],1));
y_out = gpuArray(single(zeros(Ny*Nz*Nt,Nx,Ncoils)));
% y_out = single(zeros(Ny*Nz*Nt,Nx,Ncoils));
y_out(mask(:,1),:,:) = y;
y = reshape(y_out,Ny,Nz,Nt,Nx,Ncoils);
clear y_out
% if gpu_flag
%     y = gpuArray(y);
% end
end

