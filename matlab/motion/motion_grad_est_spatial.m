% spatial handling of 20251022_Nan_fsm17

function [est_out] = motion_grad_est_spatial(motion_str, timing_str,init_str)

%%
kspace_cart_orig_fid = motion_str.kspace_cart_orig_fid;
ratios               = motion_str.ratios;
a_nav_refs           = motion_str.a_nav_refs;
Bas                  = motion_str.Bas;
mask_B0_nav          = motion_str.mask_B0_nav;
SEs_perm             = motion_str.SEs_perm;
TEs_fid              = motion_str.TEs_fid;
flip_flag            = motion_str.flip_flag;
size_nav             = motion_str.size_nav;
ky_fid               = motion_str.ky_fid;
kz_fid               = motion_str.kz_fid;

shot                 = motion_str.shot;
shot_fid             = motion_str.shot_fid;
shot_fid_in          = motion_str.shot_fid_in;
% shot_fid_gap         = motion_str.shot_fid_gap;
nav_rep              = motion_str.nav_rep;
Nrep                 = motion_str.Nrep;

nshot_total          = motion_str.nshot_total;
ratios_all           = motion_str.ratios_all;

tmp = num2cell(size_nav);
[Ny_nav, Nz_nav, Nx_nav] = tmp{:};

Ncoils = size(SEs_perm,4);
clear tmp


%%
fid_idx     = timing_str.fid_idx;
fid_idx_in  = timing_str.fid_idx_in;
resp_bin    = timing_str.resp_bin;
fid_int     = timing_str.fid_int;
nbins       = max(resp_bin); % should also equals to numel(fid_idx_in)



%%
m_est = zeros(nbins,6);
cr_est_poly = zeros(nbins,9);
c_est = zeros(nbins,Ncoils);
if isfield(init_str,'B0_est_spatial')
    B0_est_spatial = init_str.B0_est_spatial;
else
    B0_est_spatial = zeros([size_nav nbins]);
end
%% first, do B0 est, estimate B0 prep
if isfield(timing_str,'nt_sel_dB0') && ~isempty(timing_str.nt_sel_dB0)
    nt_sel_dB0 = timing_str.nt_sel_dB0;
else
    nt_sel_dB0 = [1  5];
end

Nt_dB0 = numel(nt_sel_dB0);

ky_sel = ky_fid(:,nt_sel_dB0);
kz_sel = kz_fid(:,nt_sel_dB0);
kx_sel = [ceil(Nx_nav/5+1):ceil(Nx_nav/5*4)];

mask_dB0_all = zeros(Ny_nav*Nz_nav,Nx_nav,1,Nt_dB0,shot_fid/shot_fid_in);

for ii_bin = 1:shot_fid/shot_fid_in
for t = 1:Nt_dB0
    tmp_idx = sub2ind([Ny_nav,Nz_nav],ky_sel((ii_bin-1)*shot_fid_in+(1:shot_fid_in),t),kz_sel((ii_bin-1)*shot_fid_in+(1:shot_fid_in),t));
    mask_dB0_all(tmp_idx,kx_sel,1,t,ii_bin) = 1;
end
end
mask_dB0_all = reshape(mask_dB0_all,Ny_nav,Nz_nav,Nx_nav,1,Nt_dB0,shot_fid/shot_fid_in);

if flip_flag
    mask_dB0_all = flip(flip(mask_dB0_all,1),2);
end

mask_dB0_all = logical(repmat(mask_dB0_all,[1 1 1 Ncoils]));

%% estimate B0
c0 = zeros(nbins,1);
c_est = zeros(nbins,Ncoils);
tic
for ii_bin = 1:nbins

    % build order
    ii_bin

    % new
    mask_tmp    = zeros(Ny_nav,Nz_nav,Nx_nav,1     ,Nt_dB0);
    kspace_data = zeros(Ny_nav,Nz_nav,Nx_nav,Ncoils,Nt_dB0);
    ii_shot = find(resp_bin == ii_bin);
    shot_per_bin = numel(ii_shot);
    ii_all = ii_shot(1)
    ii_out = floor((ii_shot(1)-1)/shot_fid)+1
    for ii = 1:shot_per_bin
        ii_all = ii_shot(ii)
        if fid_int(ii_shot(ii))==1
            for it = 1:numel(nt_sel_dB0)
                kspace_data(ky_sel(fid_idx(ii_all),it),kz_sel(fid_idx(ii_all),it),:,:,it) = fftc(permute(kspace_cart_orig_fid{ii_out}{mod(ii_all-1,shot_fid)+1}(:,nt_sel_dB0(it),:),[1 3 2]),1).*ratios_all(ii_all); % Nx,Ncoils, nt_sel
                mask_tmp   (ky_sel(fid_idx(ii_all),it),kz_sel(fid_idx(ii_all),it),kx_sel,1,it) = 1; % this doesn't contain coil
            end
        end
    end
    if flip_flag
        mask_tmp = logical(flip(flip(mask_tmp,1),2));
    end
% old
%     ii_all = (find(resp_bin == ii_bin,1))
%     ii_in = fid_idx_in(ii_bin)
%     ii_out = floor((ii_all-1)/shot_fid)+1
% 
%     mask_tmp = mask_dB0_all(:,:,:,1,:,ii_in); % this doesn't contain coil
% 
% 
%     kspace_data = zeros(Ny_nav,Nz_nav,Nx_nav,Ncoils,numel(nt_sel_dB0));
%     for ii = 1:shot_fid_in
%         for it = 1:numel(nt_sel_dB0)
%             kspace_data(ky_sel((ii_in-1)*shot_fid_in + ii,it),kz_sel((ii_in-1)*shot_fid_in + ii,it),:,:,it) = fftc(permute(kspace_cart_orig_fid{ii_out}{(ii_in-1)*shot_fid_in + ii}(:,nt_sel_dB0(it),:),[1 3 2]),1).*ratios_all(ii_all); % Nx,Ncoils, nt_sel
%         end
%     end

% both
    if flip_flag
    kspace_data = flip(flip(kspace_data,1),2);
    end

    kspace_data = permute(kspace_data,[1 2 3 5 4]); % Ny, Nz, Nx, Nt, Ncoils
%     kspace_data = comp(kspace_data);
    kspace_data = reshape(kspace_data,prod(size_nav)*Nt_dB0,Ncoils);
    
%     kspace_sig = kspace_data;
    kspace_sig = kspace_data(mask_tmp(:),:);

    % the image ref will be the one after motion
%     img_ref_b0 = a_nav_refs(:,:,:,1,nt_sel_dB0);
    img_ref_b0 = a_nav_refs(:,:,:,1,nt_sel_dB0).*exp(1i*2*pi*B0_est_spatial(:,:,:,ii_bin).*reshape(TEs_fid(nt_sel_dB0),1,1,1,1,[]));
    if ii_bin>1
        img_ref_b0 = img_ref_b0 .* exp(1i*c0(ii_bin-1)*reshape(TEs_fid(nt_sel_dB0),1,1,1,1,[]));
    end
    [~,~,~,img_ref_b0] = motion_ops(size_nav,m_est(ii_bin,:),[0 0 0],img_ref_b0);   
    img_ref_SEs = img_ref_b0.*SEs_perm;


    kspace_ref_new = fft3c(img_ref_SEs);
    kspace_ref_new = permute(kspace_ref_new,[1 2 3 5 4]); % Ny,Nz,Nx,Nt,Ncoils
    kspace_ref_new = reshape(kspace_ref_new,prod(size_nav)*Nt_dB0,Ncoils);
    kspace_ref_new = kspace_ref_new(mask_tmp(:),:);

    % build C grad
    time_vec = zeros(size(mask_tmp));
    for it = 1:Nt_dB0
        mask_tmp_dB0 = logical(zeros(size(mask_tmp)));
        mask_tmp_dB0(:,:,:,it) = mask_tmp(:,:,:,it);
        time_vec(mask_tmp_dB0(:)) = TEs_fid(nt_sel_dB0(it));
    end
    time_vec = time_vec(mask_tmp(:));
    clear mask_tmp_dB0

    C_grad = 1i* 2*pi * kspace_ref_new .* time_vec;
    C_grad_all = cat(1,real(C_grad),imag(C_grad));


    % estimate


    delta_k = kspace_sig-kspace_ref_new;
    delta_k = cat(1,real(delta_k),imag(delta_k));
    
    % per-coil estimate
    for ii_c = 1:Ncoils
    c_est(ii_bin,ii_c) = diag(pinv(C_grad_all(:,ii_c))*delta_k(:,ii_c)).';
    end
%     c(ii_bin,:) = diag(pinv(C_grad_all(:,:))*delta_k(:,:)).';
    % overall estimate
    c0(ii_bin) = diag(pinv(C_grad_all(:))*delta_k(:)).';
    toc

end

%%
B0s = zeros(prod(size_nav),nbins);
SEs_tmp = reshape(SEs_perm,[],Ncoils);
for ii = 1:prod(size_nav)
[~,ii_c] = max(abs(SEs_tmp(ii,:)));
B0s(ii,:) = c_est(:,ii_c);
end
B0s = reshape(B0s, [size_nav nbins]);

for ii_bin = 1:nbins
tmp1 = B0s(:,:,:,ii_bin);
tmp1 = tmp1(mask_B0_nav(:));
tmp2 = reshape(Bas,[],9);
tmp2 = tmp2(mask_B0_nav(:),:);
tmp = pcg(tmp2'*tmp2,tmp2'*tmp1);
cr_est_poly(ii_bin,:) = tmp.';
end

B0_est_grad = reshape(reshape(Bas,[],9)*cr_est_poly.',[size_nav,nbins]);
%% 
B0_est_spatial = B0_est_grad;
% %% just for test, 20260604, average nt 1 and 5
% for ii_shot = 1:numel(kspace_cart_orig_fid)
%     for ii_shot_in = 1:numel(kspace_cart_orig_fid{ii_shot})
%         kspace_cart_orig_fid{ii_shot}{ii_shot_in}(:,1,:) = (kspace_cart_orig_fid{ii_shot}{ii_shot_in}(:,1,:)+kspace_cart_orig_fid{ii_shot}{ii_shot_in}(:,5,:))/2;
%     end
% end
% a_nav_refs(:,:,:,1,1) = (a_nav_refs(:,:,:,1,1)+a_nav_refs(:,:,:,1,5))/2;
%% second, motion est
motion_grad = 0.2*eye(6); % this is to create motion grad
img_grad = zeros([size_nav,1,6]);

if isfield(timing_str,'nt_sel_mo') && ~isempty(timing_str.nt_sel_mo)
    nt_sel_mo = timing_str.nt_sel_mo;
else
    nt_sel_mo = [1  3];
end
%% motion and dB0 estimation from real kspace

ky_sel = ky_fid(:,nt_sel_mo);
kz_sel = kz_fid(:,nt_sel_mo);
% kx_sel = [ceil(Nx_nav/5+1):ceil(Nx_nav/5*2),ceil(Nx_nav/5*3+1):ceil(Nx_nav/5*4)];
kx_sel = [ceil(Nx_nav/5+1):ceil(Nx_nav/5*4)];

Nt_mot = numel(nt_sel_mo);
mask_mot_all = zeros(Ny_nav*Nz_nav,Nx_nav,1,Nt_mot,shot_fid/shot_fid_in);
% shot_fid_in = 4;
for ii_bin = 1:shot_fid/shot_fid_in
for t = 1:Nt_mot
    tmp_idx = sub2ind([Ny_nav,Nz_nav],ky_sel((ii_bin-1)*shot_fid_in+(1:shot_fid_in),t),kz_sel((ii_bin-1)*shot_fid_in+(1:shot_fid_in),t));
   
    mask_mot_all(tmp_idx,kx_sel,1,t,ii_bin) = 1;
end
end
mask_mot_all = reshape(mask_mot_all,Ny_nav,Nz_nav,Nx_nav,1,Nt_mot,shot_fid/shot_fid_in);
if flip_flag
    mask_mot_all = flip(flip(mask_mot_all,1),2);
end


mask_mot_all = logical(repmat(mask_mot_all,[1 1 1 Ncoils]));



%% move to GPU
SEs_perm = gpuArray(SEs_perm);



%% signal
tic;
% kspace_ref_new = kspace_ref;
m_est = zeros(nbins,6);

for ii_bin = 1:nbins

    % build order
    ii_bin

    % new
    mask_tmp    = zeros(Ny_nav,Nz_nav,Nx_nav,Ncoils,Nt_mot);
    kspace_data = zeros(Ny_nav,Nz_nav,Nx_nav,Ncoils,Nt_mot);
    ii_shot = find(resp_bin == ii_bin);
    shot_per_bin = numel(ii_shot);

    for ii = 1:shot_per_bin
        ii_all = ii_shot(ii)
        ii_out = floor((ii_all-1)/shot_fid)+1
        ii_in  = mod(ii_all-1,shot_fid)+1
        if fid_int(ii_shot(ii))==1
            for it = 1:numel(nt_sel_mo)
                kspace_data(ky_sel(fid_idx(ii_all),it),kz_sel(fid_idx(ii_all),it),:,:,it) = fftc(permute(kspace_cart_orig_fid{ii_out}{ii_in}(:,nt_sel_mo(it),:),[1 3 2]),1).*ratios_all(ii_all); % Nx,Ncoils, nt_sel
                mask_tmp(ky_sel(fid_idx(ii_all),it),kz_sel(fid_idx(ii_all),it),kx_sel,:,it) = 1; % this doesn't contain coil
            end
        end
    end
    if flip_flag
        mask_tmp = logical(flip(flip(mask_tmp,1),2));
    end

    % old
%     ii_all = (find(resp_bin == ii_bin,1))
%     ii_in = fid_idx_in(ii_bin)
%     ii_out = floor((ii_all-1)/shot_fid)+1
% 
%     mask_tmp = mask_mot_all(:,:,:,:,:,ii_in);


%     kspace_data = zeros(Ny_nav,Nz_nav,Nx_nav,Ncoils,Nt_mot);
%     for ii = 1:shot_fid_in
%     for it = 1:numel(nt_sel)
%         kspace_data(ky_sel((ii_in-1)*shot_fid_in + ii,it),kz_sel((ii_in-1)*shot_fid_in + ii,it),:,:,it) = fftc(permute(kspace_cart_orig_fid{ii_out}{(ii_in-1)*shot_fid_in + ii}(:,nt_sel(it),:),[1 3 2]),1).*ratios_all(ii_all); % Nx,Ncoils, nt_sel
%     end
%     end

% both
    
    if flip_flag
    kspace_data = flip(flip(kspace_data,1),2);
    end

    % update grad
    if ii_bin>1
        img_ref_new = a_nav_refs(:,:,:,:,nt_sel_mo).*exp(1i*B0_est_spatial(:,:,:,ii_bin-1).*reshape(TEs_fid(nt_sel_mo),1,1,1,1,[])); % add B0 effect
        img_ref_new = gpuArray(img_ref_new);

        [~,~,~,img_ref_new] = motion_ops(size_nav,m_est(ii_bin-1,:),[0 0 0],img_ref_new);
        
        img_ref_SEs = img_ref_new.*SEs_perm;
    else
        img_ref_new = a_nav_refs(:,:,:,:,nt_sel_mo).*exp(1i*B0_est_spatial(:,:,:,1).*reshape(TEs_fid(nt_sel_mo),1,1,1,1,[])); % add B0 effect
        img_ref_new = gpuArray(img_ref_new);
         
        img_ref_SEs = img_ref_new.*SEs_perm;
    end

    img_grad = zeros([size_nav,1,numel(nt_sel_mo),6]);
    for ii = 1:6
    [~,~,~,img_grad(:,:,:,1,:,ii)] = motion_ops(size_nav,motion_grad(ii,:),[0 0 0],img_ref_new); % Ny,Nz,Nx,forNcoils, Nt, Naxis
    end
    
    img_grad_SEs = img_grad.*SEs_perm;
    
    % updatebuild grad matrix M_grad
    M_grad = zeros(numel(kx_sel)*shot_per_bin*Ncoils*numel(nt_sel_mo),6);
    for ii = 1:6
        tmp = fft3c(img_grad_SEs(:,:,:,:,:,ii))-fft3c(img_ref_SEs);
        tmp = tmp(mask_tmp(:));
        M_grad(:,ii) = tmp/0.2;
    end

    M_grad_all = cat(1,real(M_grad),imag(M_grad));
  
    
    
%     kspace_sig = comp(kspace_data);
    kspace_sig =kspace_data;
    kspace_sig = kspace_sig(mask_tmp(:));

    kspace_ref_new = fft3c(img_ref_SEs);
    kspace_ref_new = kspace_ref_new(mask_tmp(:));
    
    
    % estimate
    
    
    delta_k = kspace_sig-kspace_ref_new;
    delta_k = cat(1,real(delta_k),imag(delta_k));

    if ii_bin ==1
    
        m_est(ii_bin,:) = (pinv(M_grad_all)*delta_k).';
    else
        m_est(ii_bin,:) = m_est(ii_bin-1,:) + (pinv(M_grad_all)*delta_k).';
    end
    

    toc

end

% 
SEs_perm = gather(SEs_perm);
clear img_ref_new  kspace_ref_new img_grad* img_ref_SEs delta_k M_grad* kspace_data tmp kspace_sig
%%

est_out.m_est = m_est;
est_out.cr_est_poly = cr_est_poly;
est_out.c_est = c_est;

est_out.nt_sel_dB0 = nt_sel_dB0;
est_out.nt_sel_mo  = nt_sel_mo;
return