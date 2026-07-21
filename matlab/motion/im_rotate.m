function [out,T,Th] = im_rotate(size_img,motion_params,in, int_method)
% in: 3D image or 3D image + time
% motion_params: 1x6 row vector, first three are rotation angles in degree,
% last three are translation. The parameters are obtained from afni to
% register in to out

if nargin < 4
    int_method = 'cubic';
end
Nx = size_img(1);
Ny = size_img(2);
Nz = size_img(3); 

pos_rot=motion_params(1:3)/180*pi;
pos_tra=motion_params([5 6 4]); % Ny(front->back), Nz(left->right), Nx (head->foot);
pos_tra=pos_tra.*[-1 -1 1];

rotx = @(t) [1 0 0; 0 cos(t) -sin(t) ; 0 sin(t) cos(t)] ;
roty = @(t) [cos(t) 0 sin(t) ; 0 1 0 ; -sin(t) 0  cos(t)] ;
rotz = @(t) [cos(t) -sin(t) 0 ; sin(t) cos(t) 0 ; 0 0 1] ;


for ii_k = 1:3
    size_tmp = ones(1,3);
    size_tmp(ii_k) = size_img(ii_k);
    kGrid{ii_k} = reshape(2*pi*[-size_img(ii_k)/2:(size_img(ii_k)/2-1)]/(size_img(ii_k)),size_tmp);  % this is from -pi to pi
end

mat_trans = exp(-1i* (kGrid{1}*pos_tra(1) + kGrid{2}*pos_tra(2) + kGrid{3}*pos_tra(3)));  % translate matrix

% % begin rotation, option 1
% [X,Y,Z] = meshgrid([-Ny/2:Ny/2-1],[-Nx/2:Nx/2-1],[-Nz/2:Nz/2-1]);
% crds = cat(1,X(:).',Y(:).',Z(:).');
% % % good one too,
% % crds_new = rotz(pos_rot(1))*roty(pos_rot(3))*rotx(pos_rot(2))*crds;
% % crds_conj = rotx(-pos_rot(2))*roty(-pos_rot(3))*rotz(-pos_rot(1))*crds;
% % % good one
% crds_new = rotz(pos_rot(1))*rotx(pos_rot(2))*roty(pos_rot(3))*crds;
% crds_conj = roty(-pos_rot(3))*rotx(-pos_rot(2))*rotz(-pos_rot(1))*crds;  % NW, 20240819

% NW,20240820, a different version, option 2
[X,Y,Z] = meshgrid([-Ny/2:Ny/2-1],[-Nx/2:Nx/2-1],[-Nz/2:Nz/2-1]);
crds = cat(1,Y(:).',X(:).',Z(:).');
crds_new = rotz(-pos_rot(1))*roty(-pos_rot(2))*rotx(-pos_rot(3))*crds;
crds_conj = rotx(pos_rot(3))*roty(pos_rot(2))*rotz(pos_rot(1))*crds;  % NW, 20240819

crds = permute(reshape(crds,3,Nx,Ny,Nz),[2 3 4 1]);
crds_new = permute(reshape(crds_new,3,Nx,Ny,Nz),[2 3 4 1]);
crds_conj = permute(reshape(crds_conj,3,Nx,Ny,Nz),[2 3 4 1]);

%% build motion operator
T  = @(x)T_motion_for(x,crds,crds_new,mat_trans, int_method);
Th = @(x)T_motion_adj(x,crds,crds_conj,mat_trans, int_method);

%% do correction

if nargin>2
    
    out = T(in);
else
    out = single(zeros(size_img));
end


return
%% motion forward
function [out] = T_motion_for(in, crds, crds_new,mat_trans, int_method)
Nt = size(in,4);
for ii_t = 1:Nt  
    %     % option 1
%     out(:,:,:,ii_t) = interp3(crds(:,:,:,1),crds(:,:,:,2),crds(:,:,:,3),in(:,:,:,ii_t),crds_new(:,:,:,1),crds_new(:,:,:,2),crds_new(:,:,:,3),int_method);
%     % option 2
    out(:,:,:,ii_t) = interp3(crds(:,:,:,2),crds(:,:,:,1),crds(:,:,:,3),in(:,:,:,ii_t),crds_new(:,:,:,2),crds_new(:,:,:,1),crds_new(:,:,:,3),int_method);
    tmp = out(:,:,:,ii_t);
    tmp(isnan(tmp)) = 0;
    tmp = fft2c(fftc(tmp,3));
    tmp = tmp.*mat_trans;
    tmp = ifft2c(ifftc(tmp,3));
    out(:,:,:,ii_t) = tmp;
end
return

%% motion hermitian
function [out] = T_motion_adj(in, crds, crds_conj,mat_trans, int_method)
Nt = size(in,4);
for ii_t = 1:Nt  
    tmp = in(:,:,:,ii_t);
    
    tmp = fft2c(fftc(tmp,3));
    tmp = tmp.*conj(mat_trans);
    tmp = ifft2c(ifftc(tmp,3));
    %     % option 1
%     tmp =
%     interp3(crds(:,:,:,1),crds(:,:,:,2),crds(:,:,:,3),tmp,crds_conj(:,:,:,1),crds_conj(:,:,:,2),crds_conj(:,:,:,3),int_method);
%     % option 2
    tmp = interp3(crds(:,:,:,2),crds(:,:,:,1),crds(:,:,:,3),tmp,crds_conj(:,:,:,2),crds_conj(:,:,:,1),crds_conj(:,:,:,3),int_method);
    tmp(isnan(tmp)) = 0;
    out(:,:,:,ii_t) = tmp;
end
return