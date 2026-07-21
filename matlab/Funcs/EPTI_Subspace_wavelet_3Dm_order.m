function [im_recon,alpha,history]=EPTI_Subspace_wavelet_3Dm_order(kdata,mask_sample,smap,Phase_total,Phi,r_ops,Func,Nstep, a_init,callback, motion_opt)
% NW, 20240417, wavelet for motion, 3D recon
% 2024.08.18, change the order of motion and phase. this version, phase first,
% then motion operator
% kdata: dim: number of time points in order of mask_sample, Nx,Ncoils
% mask_sample: dim: Ny,NZ,Nt,Nx,1,nbins
% smap: dim: Ny,Nz,1,Nx,Ncoils;
% Phase_total: dim: Ny,Nz,Nt,Nx,nbins
% Func: a string indicating which regularization function to use. Options:
%       spatial TV, LLR
% Nstep: how many intermediate x to save
% a_init: initial guess
% callback: call back functions

% Tikhonov
if nargin < 7
    a_init = [];
    callback = false;
    Nstep = 1;
    Func = 'wavelet';
elseif nargin < 8
    a_init = [];
    callback = false;
    Nstep = 1;
elseif nargin < 9
    a_init = [];
    callback = false;
elseif nargin < 10
    callback = false;
end
% Tikhonov
[~,nx,~] = size(kdata);
[ny,nz,~,nx,ncoil] = size(smap);

motion_params = motion_opt.motion_params;
pad_num = motion_opt.pad_num;

ns1 = size(Phase_total,5); % number of shot with different B0
ns2 = size(motion_params,1); 
if ns1~=ns2 && (ns1>1 && ns2>1)
    error('bin dimension mismatch')
end
ns = max(ns1,ns2);

K=size(Phi,2);
T = size(Phi,1)
size_img = [ny,nz,nx];  % exclude temporal /coefficient dimension
% Subspace basis projection
dims=[ny,nz,nx,K];
dims_adj=[ny,nz,nx,T];

prep = @(a)reshape(a,ny,nz,nx,K);
prep_adj = @(y)reshape(y,ny,nz,nx,[]);

vec = @(x)x(:);

[X,Y] = meshgrid([-nz/2:nz/2-1],[-ny/2:ny/2-1]);
% masks = X.^2+Y.^2<(min(ny,nz)^2/4);
masks = ones(ny,nz);


% if ns==1
% 
%     if sum(motion_params==0) == 6  % all 6 motion parameters are 0
%         Tm_for = @(a)a;
%         Tm_adj = @(y)y;
%     else
%         [~,Tm_for, Tm_adj,~] = motion_ops(size_img, motion_params,pad_num);
% %         [~,Tm_for, Tm_adj] = im_rotate(size_img, motion_params);  % NW, 20240401, test
%     end
% 
%     T_for = @(a) temporal_forward(a, Phi, dims);  % output: Ny,Nz,Nx,Nt
%     T_adj = @(x) temporal_adjoint(x, Phi,dims_adj);   % output: Ny,Nz,Nx,K
%     % Image Phase
%     % Phase_total=Phase_T.*repmat(Phase0,[1,1,nt]);
%     P_for = @(x) bsxfun(@times, x, Phase_total); % here the output dimension is Ny,Nz,Nx,Nt
%     P_adj = @(y) bsxfun(@times, y, conj(Phase_total));  % here the output dimension is Ny,Nz,Nx,Nt
%     
%     % permute image dimension
%     Perm_for = @(a) permute(a,[1 2 4 3]); % here the output dimension is Ny,Nz,Nt,Nx
%     Perm_adj = @(y) ipermute(y,[1 2 4 3]); % here the output dimension is Ny,Nz,Nx,,Nt
%     % Sampling mask
%     % smap=repmat(smap,[1 1 1 nt]);
%     S_for = @(a) bsxfun(@times, smap, a);  % here output is Ny,Nz,Nt,Nx,Ncoils
% %     S_adj = @(as) squeeze(sum(bsxfun(@times, conj(smap), as), 5));  % output dimension is Ny,Nz,Nt,Nx
%     S_adj = @(as) sum(bsxfun(@times, conj(smap), as), 5);  % output dimension is Ny,Nz,Nt,Nx
%     % Sampling mask
%     F_for = @(x) fft2c(x);  % Nx dim is already in image space
%     F_adj = @(y) ifft2c(y);
%     % Sampling mask
%     M_for = @(a) vec(Mask_3D(reshape(a,[],nx,ncoil), mask_sample));  % output dimension: Nshot, Nx,Ncoils
%     M_adj = @(y) Mask_3D_adj(reshape(y,[],nx,ncoil), mask_sample); % output dimension, Ny,Nz,Nt,Nx,Ncoils
% 
%     A_for = @(a) M_for(F_for(S_for(Perm_for(Tm_for(P_for(T_for(prep(a))))))));
%     A_adj = @(y) T_adj(P_adj(Tm_adj(prep_adj(Perm_adj(S_adj(F_adj(M_adj(y))))))));
% 
%     AHA = @(a) vec(A_adj(A_for(a)));
%     AHA_lsqr = @(a,lFlag)A_adj(A_for(a));
% else  % ns >1, more than one status
    T_for = @(a) temporal_forward(a, Phi, dims);
    T_adj = @(x) temporal_adjoint(x, Phi, dims_adj);

    S_for = @(a) bsxfun(@times, smap, a);  % here output is Ny,Nz,Nt,Nx,Ncoils
%     S_adj = @(as) squeeze(sum(bsxfun(@times, conj(smap), as), 5));  % output dimension is Ny,Nz,Nt,Nx
    S_adj = @(as) sum(bsxfun(@times, conj(smap), as), 5);  % output dimension is Ny,Nz,Nt,Nx
       % Sampling mask
    F_for = @(x) fft2c(x);  % Nx dim is already in image space
    F_adj = @(y) ifft2c(y);

    % permute image dimension
    Perm_for = @(a) permute(a,[1 2 4 3]); % here the output dimension is Ny,Nz,Nt,Nx
    Perm_adj = @(y) ipermute(y,[1 2 4 3]); % here the output dimension is Ny,Nz,Nx,,Nt
    
    l_ts(1) = 0;    % this is the begining of the first segmentation

    for ii_bin = 1:ns
        l_ts(ii_bin+1) = l_ts(ii_bin)+numel(find(mask_sample(:,:,:,1,1,ii_bin)==1)); % this is the endpoint of this shot
        l_bg = l_ts(ii_bin)+1; % bg point for current segment
        l_end = l_ts(ii_bin+1); % end point
        % Image Phase
        % Phase_total=Phase_T.*repmat(Phase0,[1,1,nt]);
        P_for = @(x) bsxfun(@times, x, Phase_total(:,:,:,:,ii_bin)); % here the output dimension is Ny,Nz,Nt,Nx
        P_adj = @(y) bsxfun(@times, y, conj(Phase_total(:,:,:,:,ii_bin)));  % here the output dimension is Ny,Nz,Nx,Nt
        
        if sum(motion_params(ii_bin,:)==0) == 6   % all 6 motion parameters are 0
            Tm_for = @(a)a;
            Tm_adj = @(y)y;
        else
            [~,Tm_for, Tm_adj,~] = motion_ops(size_img, motion_params(ii_bin,:),pad_num);
%             [~,Tm_for, Tm_adj] = im_rotate(size_img, motion_params(ii_bin,:));  % NW, 20240401, test
        end
        % Sampling mask
        M_for = @(a) Mask_3D(reshape(a,[],nx,ncoil), mask_sample(:,:,:,:,:,ii_bin));  % output dimension: Nshot, Nx,Ncoils
        M_adj = @(y) Mask_3D_adj(reshape(y(l_bg:l_end,:,:),[],nx,ncoil), mask_sample(:,:,:,:,:,ii_bin)); % output dimension, Ny,Nz,Nt,Nx,Ncoils
        
        if ii_bin == 1
            A_for = @(a) M_for(masks.*F_for(S_for(Perm_for(Tm_for(P_for(T_for(prep(a))))))));
            A_adj = @(y) T_adj(P_adj(Tm_adj(prep_adj(Perm_adj(S_adj(F_adj(M_adj(y).*masks)))))));
        else
            A_for = @(a) cat( 1, A_for(a), M_for(masks.*F_for(S_for(Perm_for(Tm_for(P_for(T_for(prep(a)))))))) );
            A_adj = @(y) A_adj(y) + T_adj(P_adj(Tm_adj(prep_adj(Perm_adj(S_adj(F_adj(M_adj(y).*masks)))))));
        end

    end
    A_for = @(a)A_for(a);
    AHA = @(a) vec(A_adj(A_for(a)));
% end

%% scaling
tmp = dimnorm(ifft2c(kdata), 3);
tmpnorm = dimnorm(tmp, 4);
tmpnorm2 = sort(tmpnorm(:), 'ascend');
% match convention used in BART
p100 = tmpnorm2(end);
p90 = tmpnorm2(round(.9 * length(tmpnorm2)));
p50 = tmpnorm2(round(.5 * length(tmpnorm2)));
if (p100 - p90) < 2 * (p90 - p50)
    scaling = p90;
else
    scaling = p100;
end

% scaling=1;
fprintf('\nScaling: %f\n\n', scaling);

kdata = kdata ./ scaling;
ksp_adj = reshape(A_adj(kdata),dims);
kdata = gather(kdata);
%%  wavelet setup
wlev = 4;  % wavelet level
wname = 'sym4'; % wavelet name
padmat = ceil([ny nz nx]/2^wlev)*2^wlev-[ny nz nx]; % each dimension need to be divisible by 2^wlet, which is 16 if wlet=4

if sum(padmat)~=0 % if padding
    pad = @(x)padarray(prep(x),[padmat 0],0, 'post');
    crop = @(x)x(1:ny, 1:nz, 1:nx,:);
    W = @(x,p)wave3d(pad(x.*exp(-1i*p)),wlev, wname, r_ops.mask);
    Wh = @(x,p)vec(crop(iwave3d(x))).*exp(1i*p(:));
else
    W = @(x,p)wave3d(prep(x.*exp(-1i*p)),wlev, wname, r_ops.mask);
    Wh = @(x,p)vec(iwave3d(x)).*exp(1i*p(:));
end
WhW = @(x,p)vec(x);
figmake = @(WX)wavelet_figmake(WX);

objfun = @(a, Wx, lam) norm_mat(kdata(:) - vec(A_for(a)))^2 + lam*sum(abs(vec(wavelet_collect(Wx))));
r_ops.dc = @(a) norm_mat(kdata(:) - vec(A_for(a)))^2;
r_ops.reg = @(Wx)sum(abs(vec(wavelet_collect(Wx))));

x = a_init ./ scaling;
p = angle(x);
% p = zeros(size(x));
Wx = W(x,p);
im0 = complex(figmake(Wx));

lambda = r_ops.lambda;
rho = r_ops.rho;
max_iter = r_ops.iter;
alpha = lambda/rho;
% define admm variable u = y/rho;
u = wavelet_applyfun(@(x)zeros(size(x)),Wx);
z = wavelet_applyfun(@(x)zeros(size(x)),Wx);

ABSTOL = 1e-4;
RELTOL = 1e-2;

abserr = sqrt(numel(ksp_adj)) * ABSTOL;

fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
    'lsqr iters', 'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
tic
for ii = 1:max_iter
%     figure(100),imshow([log(complex(im0)), log(complex(figmake(Wx)))],[]),drawnow;
%     figure(101),imshow(abs(10*[im0,figmake(Wx)]/max(im0(:)))),drawnow
    x_old = x;
    z_old = z;
    z = wavelet_applyop(@plus, Wx, u);
    z = wavelet_applyop(@times, wavelet_applyfun(@(x)sign(x),z), wavelet_applyfun(@(x)max(abs(x)-alpha,0),z)); % soft threshold
    u = wavelet_applyop(@plus, u, wavelet_applyop(@minus, Wx,z));
    [x,~,~,nitr] = pcg(@(x)AHA(x)+rho/2*WhW(x,p), vec(ksp_adj)+rho/2*Wh(wavelet_applyop(@minus,z,u),p),...
            [],median([ii, 5, max_iter]), [],[],x(:));
    % record

    p = angle(x);
%     p = zeros(size(x));
    Wx = W(x,p);
    history.objval(ii) = objfun(x, Wx, lambda);
    history.lsqr_nitr(ii) = nitr;
    history.r_norm(ii) = norm_mat(wavelet_collect(wavelet_applyop(@minus,Wx,z)));
    history.s_norm(ii) = norm_mat(-rho * wavelet_collect(wavelet_applyop(@minus,z, z_old)));
    history.eps_pri(ii) = abserr + RELTOL * max(norm(wavelet_collect(Wx)), norm(wavelet_collect(z)));
    history.eps_dual(ii) = abserr + RELTOL * norm_mat(rho * wavelet_collect(u));
    if Nstep>1 && mod(ii, Nstep)==0
        history.intermx(:,round(ii/Nstep)) = x(:);
    end
    history.dc = r_ops.dc(x);
    history.reg = r_ops.reg(Wx);
    fprintf('%3d\t%10d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', ii, ...
        sum(history.lsqr_nitr), history.r_norm(ii), history.eps_pri(ii), ...
        history.s_norm(ii), history.eps_dual(ii), history.objval(ii));

end

toc;
disp(' ');
disp('Reconstruction Done');

%% Project and re-scale
alpha = x;
% res_a=reshape(res,dims);
im_recon=temporal_forward(alpha(:), Phi, dims);
% im = T_for(alpha);

disp('Rescaling')
alpha = alpha * scaling;
try
history.intermx = history.intermx * scaling;
end
im_recon = im_recon * scaling;


