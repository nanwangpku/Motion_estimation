%% this is a test code for recon multi-echo GRE with motion
%% setup path
addpath(genpath('./matlab'))
%% load data
load(['.',filesep,'data',filesep,'example_corrupted_data_meGRE2.mat'])  % raw multi-echo GRE data
load(['.',filesep,'data',filesep,'motion_estimation.mat'])  % output of motion_bkm_main.m / motion_bkm_python.py:
                                                              % motion_est, cr_est, Bas_norm, voxel_size, size_nav,
                                                              % fid_sel, fid_dis, nshot_total

%% set up image parameters
% Nx = 240;  % SI
% Ny = 240;  % AP
% Nz = 216;  % RL
[Ny,Nz,Nx,Ncoils] = size(SEs)
Nt = numel(TEs)
SEs = reshape(SEs,Ny,Nz,1,Nx,Ncoils);
%% set up flag
filt_flag = 1; % hamming window flag to k-space for high-SNR outcome given motion
flip_flag = 1;
useGPU = 1;
%%
Func = 'wavelet';
r_ops.dim = 3;

Nstep = 1;
r_ops.iter = 10;
r_ops.mask = logical(ones(Ny,Nz,Nx));
r_ops.lambda = .003;

r_ops.rho = r_ops.lambda;
r_ops.max_iter = 10;
r_ops.tol = 1e-6;

%% motion cluster, this is very important to reduce recon time
nbins = 13; % this is a practical value given the motion profile; can be changed due to the need. For recon including all motion status, please check out the work and github path
[IDX, motion_params] = kmeans(motion_est .*[1 1 1 voxel_size],nbins);  % translation should be mutiplied with voxel size difeerence between SEMAN-nav and actual data
B0_est = zeros(Ny,Nz,Nx,nbins);
for ii_bin = 1:nbins;
    B0_est(:,:,:,ii_bin) = imresize3(reshape(Bas_norm * mean(cr_est(IDX==ii_bin,:),1).',size_nav),[Ny,Nz,Nx]);  % 2nd order is smooth, can do a direct interpolation
end

resp_bin = interp1([1:(fid_sel+fid_dis):nshot_total], IDX, 1:nshot_total, 'nearest','extrap');

%% generate mask

mask_motion = zeros(Ny*Nz,1,Nx,1,nbins);
for ii_bin = 1:nbins
    k_idx = find(resp_bin==ii_bin);
    ky_tmp = ky(k_idx);
    kz_tmp = kz(k_idx);
    k_tmp = sub2ind([Ny0,Nz0],ky_tmp,kz_tmp);
    mask_motion(k_tmp,1,:,1,ii_bin) = 1;
end
mask_motion = reshape(mask_motion,Ny,Nz,1,Nx,1,nbins);
mask_motion = logical(mask_motion);

mask_all = zeros(Ny*Nz,1,Nx);

k_tmp = sub2ind([Ny,Nz],ky,kz);
mask_all(k_tmp,1,:) = 1;
mask_all = reshape(mask_all,Ny,Nz,1,Nx,1,1);
mask_all = logical(mask_all);

if flip_flag
    mask_motion = flip(flip(mask_motion,2),1);
    mask_all = flip(flip(mask_all,2),1);
end

%%

pad_num = [10 9 10];  % pad in k-space, for improved image quality when there is rotation
motion_opt.motion_params = motion_params;
motion_opt.pad_num = pad_num;

motion_opt_noco.motion_params = zeros(1,6);
motion_opt_noco.pad_num = [0 0 0];


L = 1;
Phi = ones(1,1);  % this is not usable for this case, only usable in EPTI recon


if exist('useGPU','var') && useGPU

    mask_motion = gpuArray(single(mask_motion));
    mask_all = gpuArray(single(mask_all));
    Phi = gpuArray(single(Phi));

    SEs = gpuArray(single(SEs));
end

% windowing

wind_x = hamming_flat(Nz,ceil(Nz*4/15));
wind_y = hamming_flat(Ny,ceil(Ny*1/8));
wind_z = hamming_flat(Nx,ceil(Nx*1/8));
[X,Y,Z] = meshgrid(wind_x,wind_y,wind_z);
M_soft =X.*Y.*Z;
%%
%         tmps = vec([2:5:Nt;3:5:Nt;4:5:Nt;5:5:Nt]).';
a_noco = single(zeros(Ny,Nz,Nx,Nt));
a_moco = single(zeros(Ny,Nz,Nx,Nt));
%%
for t = [1 7 14]  % only do a few representative echos in this example for a quick test
    %         for t = [1 7]
    t

    kspace_tmp = permute(kspace_cart(:,:,:,:,t),[2 3 1 4]);  % Ny,Nz,Nx,Ncoils
    if flip_flag
        kspace_tmp = flip(flip(kspace_tmp,2),1);
    end
    kspace_tmp = reshape(kspace_tmp,[],Nx,Ncoils);
    mask_tmp = logical(reshape(mask_all(:,:,:,Nx,1,1),[],1));
    kspace_inp = kspace_tmp(mask_tmp(:,1),:,:);
    % no correction
    phs_noco = ones(Ny,Nz,Nx);

    [~,a,~] = EPTI_Subspace_wavelet_3Dm_order(kspace_inp,mask_all,SEs,phs_noco,Phi,r_ops,Func,Nstep, zeros(Ny,Nz,Nx,L),1, motion_opt_noco);
    a_noco(:,:,:,t) = reshape(gather(a),Ny,Nz,Nx);
    clear a

    phs = exp(1i*2*pi*reshape(B0_est(:,:,:,1:nbins),Ny,Nz,Nx,1,nbins).*reshape(TEs(t),1,1,1,[]));  % Ny,Nz,Nx,Nt,nbins
    %             phs = ones(Ny,Nz,Nx,1,nbins);
    kspace_tmp = permute(kspace_cart(:,:,:,:,t),[2 3 1 4]);
    if flip_flag
        kspace_tmp = flip(flip(kspace_tmp,2),1);
    end
    kspace_tmp = kspace_tmp.*M_soft;
    kspace_tmp = reshape(kspace_tmp,[],Nx,Ncoils);
    for ii_bin = 1:nbins
        tmp = kspace_tmp;
        mask_tmp = logical(reshape(mask_motion(:,:,:,Nx,1,ii_bin),[],1));
        tmp = tmp(mask_tmp(:,1),:,:);
        if ii_bin==1
            kspace_inp = tmp;
        else
            kspace_inp = cat(1,kspace_inp, tmp);
        end
    end
    if exist('useGPU','var') && useGPU
        phs = gpuArray(single(phs));
        kspace_inp = gpuArray(single(kspace_inp));
    end
    tic
    %             [~,a,~] = EPTI_Subspace_CG_3D_motion_order(kspace_inp,mask_sample,SEs_perm,phs,Phi,zeros(Ny,Nz,Nx,L),40,motion_opt);
    [~,a,~] = EPTI_Subspace_wavelet_3Dm_order(kspace_inp,mask_motion,SEs,phs,Phi,r_ops,Func,Nstep, zeros(Ny,Nz,Nx,L),1, motion_opt);
    toc
    a_moco(:,:,:,t) = reshape(gather(a),Ny,Nz,Nx);
    clear a
end

if exist('useGPU','var') && useGPU
    kspace_inp = gather(kspace_inp);
    Phi = gather(Phi);
    phs = gather(phs);
    mask_motion = gather(mask_motion);
    SEs = gather(SEs);

end
