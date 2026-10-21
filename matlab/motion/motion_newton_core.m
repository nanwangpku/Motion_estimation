function out = motion_newton_core(cfg)
%% motion_newton_core  Shared cirEPTI motion/B0 estimation pipeline.
%
% out = motion_newton_core(cfg)
%
% Required cfg fields:
%   data_path      - full path to dataset directory (trailing slash)
%   pos_path       - relative path to main recon .mat
%   pos_fid_path   - relative path to FID .mat
%   scout_path     - relative path to scout .mat
%   full_path      - relative path to SEs_new .mat (fits_60_NC*.mat)
%   Nt_scout_in    - scalar (e.g. 6) or vector (e.g. 1:6 or [1:6 27:32 55:60])
%
% Optional cfg fields (with defaults):
%   pos_nav_path          - '' (non-empty → split-struct loading for 260xxx)
%   scout_path2           - '' (non-empty → load temp_sp_pad from this path)
%   Nt_ahead              - 7
%   bin_center            - 1
%   nt_pick               - 6
%   coil_sel              - 1:32
%   shot_fid_in           - [] (empty=5D Gen B masks; 4=6D Gen C masks)
%   shot_fid_gap          - 1  how much gap, 1 means no gap, 2 means pick one group every two groups
%   scout_rep_sel         - 3 (for a0_all_rep non-bkfs loading)

%% Defaults
if ~isfield(cfg,'no_nav'),                   cfg.no_nav = false; end
if ~isfield(cfg,'scout_path_internal'),      cfg.scout_path_internal = ''; end
if ~isfield(cfg,'pos_nav_path'),             cfg.pos_nav_path = ''; end
if ~isfield(cfg,'Nt_ahead'),                 cfg.Nt_ahead = 7; end
if ~isfield(cfg,'bin_center'),               cfg.bin_center = 1; end
if ~isfield(cfg,'nt_pick'),                  cfg.nt_pick = 3; end
if ~isfield(cfg,'coil_sel'),                 cfg.coil_sel = 1:32; end
if ~isfield(cfg,'shot_fid_in'),              cfg.shot_fid_in = 4; end
if ~isfield(cfg,'shot_fid_gap'),             cfg.shot_fid_gap = 1; end
if ~isfield(cfg,'scout_rep_sel'),            cfg.scout_rep_sel = 2; end

%% Normalize Nt_nav → always a vector index
Nt_scout_in = cfg.Nt_scout_in;
if isscalar(Nt_scout_in)
    nt_scout_idx = 1:Nt_scout_in;
else
    nt_scout_idx = Nt_scout_in;
end
Nt_scout = numel(nt_scout_idx);

%% Fixed geometry

L   = 3;
vec = @(x)x(:);

dataset_path   = cfg.dataset_path;
bin_center  = cfg.bin_center;
nt_pick     = cfg.nt_pick;
Nt_ahead    = cfg.Nt_ahead;
shot_fid_in = cfg.shot_fid_in;
shot_fid_gap = cfg.shot_fid_gap;
coil_sel    = cfg.coil_sel;
useGPU      = false;

%% Data loading
if cfg.no_nav
    %--- No nav acquired (nav is not critical to motion estimation) ----
    % Load only the main + fid files; treat as a single "bin" (nav_rep=1)
    % since there is no separate per-bin navigator data to bin by.
    pos_data     = load(fullfile(dataset_path, cfg.pos_path), ...
       'ScanType', 'SEs','Nx','Ny','Nz','shot','Nrep','flags');
    pos_fid_data = load(fullfile(dataset_path, cfg.pos_fid_path), ...
        'kspace_cart_orig_all','ky','kz','shot','TEs');

    kspace_cart_orig_fid = pos_fid_data.kspace_cart_orig_all;
    file_name    = regexp(cfg.pos_path, filesep, 'split'); file_name = file_name{1};
    ky_fid       = pos_fid_data.ky;
    kz_fid       = pos_fid_data.kz;
    shot_fid     = pos_fid_data.shot;
    Nrep         = pos_data.Nrep;
    shot         = pos_data.shot;
    flip_flag    = pos_data.flags.flip_flag;
    B0_nav       = [];
    TEs_nav      = [];
    a0_nav       = [];
    Phi_nav      = [];
    nav_rep      = 1;
    nav_int      = [];
    is_split     = true; % unused: a_nav_sense build is skipped below
elseif isempty(cfg.pos_nav_path)
    %--- Standard: single pos_data file --------------------------------
    load(fullfile(dataset_path, cfg.pos_fid_path), 'kspace_cart_orig_fid');

    pos_data = load(fullfile(dataset_path, cfg.pos_path), ...
        'ScanType','nav_type','nav_int','*_flag', ...
        'TEs*','TEs_nav','TEs_fid', ...
        'ky_fid','kz_fid',...
        'shot','shot_fid','nrep', ...
        'Nx','Ny','Nz','SEs','emaps', ...
        'PD_rec*','T2s_rec*', ...
        'B0_nav','a0_nav','L','Phi*','phase_diff0','k*');

    file_name   = regexp(cfg.pos_path, filesep, 'split'); file_name = file_name{1};
  
    ky_fid      = pos_data.ky_fid;
    kz_fid      = pos_data.kz_fid;
    shot_fid    = pos_data.shot_fid;
    Nrep        = pos_data.nrep;
    shot        = pos_data.shot;
    flip_flag   = pos_data.flip_flag;
    B0_nav      = pos_data.B0_nav;
    TEs_nav     = pos_data.TEs_nav;
    a0_nav      = pos_data.a0_nav;
    Phi_nav     = pos_data.Phi_nav;
    nav_rep     = size(a0_nav,2);
    nav_int     = pos_data.nav_int;
    % Nx_nav      = pos_data.Nx_nav; 
    % Ny_nav      = pos_data.Ny_nav; 
    % Nz_nav      = pos_data.Nz_nav;
    is_split    = false;
else
    %--- Split-struct loading (260401_nmo, 260406_bkm17) ---------------
    pos_data     = load(fullfile(dataset_path, cfg.pos_path), ...
       'ScanType', 'SEs','Nx','Ny','Nz','shot','Nrep','flags');
    pos_nav_data = load(fullfile(dataset_path, cfg.pos_nav_path), ...
        'a_all_rep','Phi','B0_all','TEs','Nx','Ny','Nz');
    pos_fid_data = load(fullfile(dataset_path, cfg.pos_fid_path), ...
        'kspace_cart_orig_all','ky','kz','shot','TEs');

    kspace_cart_orig_fid = pos_fid_data.kspace_cart_orig_all;
    file_name    = regexp(cfg.pos_path, filesep, 'split'); file_name = file_name{1};
    ky_fid       = pos_fid_data.ky;
    kz_fid       = pos_fid_data.kz;
    shot_fid     = pos_fid_data.shot;
    Nrep         = pos_data.Nrep;
    shot         = pos_data.shot;
    flip_flag    = pos_data.flags.flip_flag;
    B0_nav       = pos_nav_data.B0_all;
    TEs_nav      = pos_nav_data.TEs;
    a0_nav       = pos_nav_data.a_all_rep;
    Phi_nav      = pos_nav_data.Phi;
    nav_rep      = size(a0_nav,3);
    nav_int      = [];
    % Nx_nav       = pos_nav_data.Nx; 
    % Ny_nav       = pos_nav_data.Ny; 
    % Nz_nav       = pos_nav_data.Nz;
    is_split     = true;
end



%% Basic image setup
[Nx, Ny, Nz, ~] = size(pos_data.SEs);
Nx_targ = Nx;             Ny_targ = Ny;             Nz_targ = Nz;

% SEs_nav_s = load(fullfile(dataset_path, cfg.full_path), 'SEs_new'); %
% this sensetivity maps may not be shifted
% SEs_nav   = SEs_nav_s.SEs_new;
SEs_nav_s = load(fullfile(dataset_path, cfg.scout_path),'SEs');
SEs_nav   = SEs_nav_s.SEs;

clear SEs_nav_s
[Nx_nav, Ny_nav, Nz_nav, Ncoils] = size(SEs_nav);

size_nav = [Ny_nav, Nz_nav, Nx_nav];

nbins   = nav_rep;

voxel_size = [Ny_targ/Ny_nav, Nz_targ/Nz_nav, Nx_targ/Nx_nav];

%% Reconstruct navigator images
if cfg.no_nav
    a_nav_sense = [];
else
    a_nav_sense = zeros(Ny_nav, Nz_nav, Nx_nav, Nt_scout, nbins);
    if ~is_split
        for ii_bin = 1:nbins
            tmp = reshape(reshape(a0_nav(:,ii_bin),[],L) * Phi_nav(nt_scout_idx,:).', ...
                Ny_nav, Nz_nav, Nx_nav, []);
            a_nav_sense(:,:,:,:,ii_bin) = tmp .* exp(1i*2*pi * B0_nav(:,:,:,ii_bin) ...
                .* reshape(TEs_nav(nt_scout_idx),1,1,1,[]));
        end
    else
        % split-struct: a0_all_rep is [Ny*Nz*L, Nx, nbins]
        for ii_bin = 1:nbins
            tmp = reshape(reshape(permute(reshape(a0_nav(:,:,ii_bin),[],L,Nx_nav),[1 3 2]),[],L) ...
                * Phi_nav(nt_scout_idx,:).', Ny_nav, Nz_nav, Nx_nav, []);
            a_nav_sense(:,:,:,:,ii_bin) = tmp .* exp(1i*2*pi * B0_nav(:,:,:,ii_bin) ...
                .* reshape(TEs_nav(nt_scout_idx),1,1,1,[]));
        end
    end
    clear tmp
end

%% Scout loading
if contains(cfg.scout_path, 'bkfs')
    disp('bkfs_scout')
    scout_data = load(fullfile(dataset_path, cfg.scout_path), 'a0_all','TEs','B0');

    scout_img = ifft3c(circshift(fft3c(scout_data.a0_all(:,:,:,nt_scout_idx)),[1 1 0 0]));
    scout_for_ratio = ifft3c(circshift(fft3c(scout_data.a0_all(:,:,:,nt_pick+Nt_ahead)),[1 1 0 0]));
    
else
    % non-bkfs with path2 (temp_sp_pad)
    scout_data  = load(fullfile(dataset_path, cfg.scout_path), 'TEs','B0');
    m = matfile(fullfile(dataset_path, cfg.scout_path));

    if ismember('a0_sp', who(m))
        disp('spatial_scout')
        scout_data2 = load(fullfile(dataset_path, cfg.scout_path),'a0_sp','L_l','Phi_all');
        scout_img = reshape(reshape(scout_data2.a0_sp(:,:,:,:,1),[],scout_data2.L_l)*scout_data2.Phi_all(nt_scout_idx,:).',Ny_nav,Nz_nav,Nx_nav,[]);
        scout_for_ratio = reshape(reshape(scout_data2.a0_sp(:,:,:,:,1),[],scout_data2.L_l)*scout_data2.Phi_all(nt_pick+Nt_ahead,:).',Ny_nav,Nz_nav,Nx_nav,[]);
        clear scout_data2
    
    else
        disp('regular_scout')
%     % non-bkfs with a0_all_rep (old style, 250618_yi)
        scout_data = load(fullfile(dataset_path, cfg.scout_path), 'a0_all','Phi','TEs','B0');
        a0_all     = scout_data.a0_all;
        temp       = permute(reshape(a0_all,Ny_nav,Nz_nav,[],Nx_nav),[1 2 4 3]);
        temp       = reshape(reshape(temp,[],L)*scout_data.Phi.', Ny_nav,Nz_nav,Nx_nav,[]);
        scout_img = temp(:,:,:,nt_scout_idx) .* exp(1i*2*pi*mean(scout_data.B0(:,:,:,:,cfg.scout_rep_sel),4) ...
                .* reshape(scout_data.TEs(nt_scout_idx),1,1,1,[]));
        scout_for_ratio = temp(:,:,:,nt_pick+Nt_ahead).* exp(1i*2*pi*mean(scout_data.B0(:,:,:,:,cfg.scout_rep_sel),4) .* scout_data.TEs(nt_pick+Nt_ahead));
    clear temp
    end
    clear m
    %     if Nt_ahead > 0
%         nf = nt_nav+Nt_ahead;
%         scout_img = temp(:,:,:,1:nf) .* exp(1i*2*pi*scout_data.B0_all(:,:,:,cfg.scout_rep_sel) ...
%             .* reshape(scout_data.TEs(1:nf),1,1,1,[]));
%     else
%         nf = Nt_nav_idx(end);
%         scout_img = temp(:,:,:,1:nf) .* exp(1i*2*pi*scout_data.B0_all(:,:,:,cfg.scout_rep_sel) ...
%             .* reshape(scout_data.TEs(1:nf),1,1,1,[]));
%     end
%     clear temp a0_all
end

TEs_fid      = scout_data.TEs(nt_scout_idx);
%% check if mask out of shoulder is needed
[mask_si, info] = detect_and_mask_shoulders(abs(scout_img(:,:,:,1)),3, 'Verbose',1,'EdgeThresh',0.35);
%%
scout_img         = scout_img .* reshape(mask_si,1,1,[]);
scout_for_ratio   = scout_for_ratio .* reshape(mask_si,1,1,[]);
if ~cfg.no_nav
    a_nav_sense   = a_nav_sense .* reshape(mask_si,1,1,[]);
end

for ii = 1:numel(kspace_cart_orig_fid) % kspace_cart_orig_fid first dim (Nx) is in image domain
    for jj = 1:numel(kspace_cart_orig_fid{ii})
        kspace_cart_orig_fid{ii}{jj} = kspace_cart_orig_fid{ii}{jj}.* mask_si;
    end
end

if ~cfg.no_nav
%% Motion estimation pass 1: scout → navigator
Nx_targ_sel = ceil(5/6*Nx_nav);
ref_frame = 2;
res_regi = [];
orient_regi = [];
deb = 0;
clear img_save motion_params

tmp = scout_for_ratio;

tmp(:,:,Nx_targ_sel:end) = 0;
img_save{1} = tmp;

for ii_bin = 1:nbins
    tmp = reshape(a_nav_sense(:,:,:,nt_pick,ii_bin), Ny_nav,Nz_nav,Nx_nav,[]);
    tmp(:,:,Nx_targ_sel:end) = 0;
    img_save{2} = tmp;

    [~,motion_params_tmp] = alignVolumes(img_save,res_regi,orient_regi,ref_frame,[3 .3],0,[],deb);
    motion_params_tmp     = squeeze(motion_params_tmp);
    T_tmp                 = motion_params_tmp(1,[4 5 6 3 1 2]);
    T_tmp(:,1:3)          = T_tmp(:,1:3)/pi*180;
    T_tmp(:,4:6)          = T_tmp(:,4:6).*[1 -1 -1].*voxel_size;
    motion_params(ii_bin,:) = T_tmp;
end
clear motion_params_tmp
motion_params_base_orig = motion_params;

%% Ratios computation (always nt_pick=6 for ratios)
disp('get ratios')

ratios = zeros(1,nbins);
for ii_bin = 1:nbins
    [~,~,~,tmp] = motion_ops([Ny_nav,Nz_nav,Nx_nav], ...
        motion_params_base_orig(ii_bin,:)./[1 1 1 voxel_size], 0, ...
        scout_for_ratio);
    ratios(ii_bin) = abs(vec(a_nav_sense(:,:,:,nt_pick,ii_bin)).') * conj(abs(tmp(:))) ...
        / (abs(vec(a_nav_sense(:,:,:,nt_pick,ii_bin)).') * conj(abs(vec(a_nav_sense(:,:,:,nt_pick,ii_bin)))));
end

%% Motion estimation pass 2: with ratios → scout → navigator
ref_frame = 2;
clear img_save motion_params

tmp = scout_for_ratio;
tmp(:,:,Nx_targ_sel:end) = 0;
img_save{1} = tmp;

for ii_bin = 1:nbins
    tmp = reshape(a_nav_sense(:,:,:,nt_pick,ii_bin),Ny_nav,Nz_nav,Nx_nav,[]) * ratios(ii_bin);
    tmp(:,:,Nx_targ_sel:end) = 0;
    img_save{2} = tmp;

    [~,motion_params_tmp] = alignVolumes(img_save,res_regi,orient_regi,ref_frame,[3 .3],0,[],deb);
    motion_params_tmp     = squeeze(motion_params_tmp);
    T_tmp                 = motion_params_tmp(1,[4 5 6 3 1 2]);
    T_tmp(:,1:3)          = T_tmp(:,1:3)/pi*180;
    T_tmp(:,4:6)          = T_tmp(:,4:6).*[1 -1 -1].*voxel_size;
    motion_params(ii_bin,:) = T_tmp;
end
clear motion_params_tmp
motion_params_base = motion_params;
else
    % No nav data: nav is not critical to motion estimation, so per-bin
    % scout<->navigator registration (motion_params_base) is skipped
    % entirely -- motion_newton_round.m only consumes it inside its
    % st_flag==1 joint-B0 block, which is itself skipped when nav_rep<2
    % (see the matching guard added there).
    disp('no nav data -- skipping scout<->navigator motion registration (motion_params_base)')
    motion_params_base = [];

    %% Ratios: the only nav-derived quantity that still matters here.
    % Two cases (nbins==1, a single scalar ratio broadcast to every shot
    % below):
    %  - an internal scout exists but the scout actually in use (cfg.scout_path)
    %    is external (bkfs): load the internal scout and use it as a stand-in
    %    single-bin "nav" to gauge signal-level mismatch against the external
    %    scout that everything else is registered to.
    %  - only the internal scout is available (no external one to compare
    %    against): there's nothing to take a ratio of, so ratios = 1.
    disp('get ratios (no-nav path)')
    if ~isempty(cfg.scout_path_internal) && contains(cfg.scout_path, 'bkfs')
        disp('using internal scout as nav substitute for ratio computation')
        % Internal scout .mat files come from recon_EPTI_3D, same as the
        % non-bkfs cfg.scout_path branch above -- auto-detect its format
        % the same way that branch does (a0_sp = current recon_EPTI_3D
        % "spatial_scout" format; a0_all/Phi = older "regular_scout" format),
        % rather than assuming one specific layout.
        mi = matfile(fullfile(dataset_path, cfg.scout_path_internal));
        if ismember('a0_sp', who(mi))
            internal_data = load(fullfile(dataset_path, cfg.scout_path_internal), 'a0_sp','L_l','Phi_all');
            a_int_for_ratio = reshape(reshape(internal_data.a0_sp(:,:,:,:,1),[],internal_data.L_l) ...
                * internal_data.Phi_all(nt_pick+Nt_ahead,:).', Ny_nav,Nz_nav,Nx_nav,[]);
        else
            internal_data = load(fullfile(dataset_path, cfg.scout_path_internal), 'a0_all','Phi','TEs','B0');
            temp_int = permute(reshape(internal_data.a0_all,Ny_nav,Nz_nav,[],Nx_nav),[1 2 4 3]);
            temp_int = reshape(reshape(temp_int,[],L)*internal_data.Phi.', Ny_nav,Nz_nav,Nx_nav,[]);
            a_int_for_ratio = temp_int(:,:,:,nt_pick+Nt_ahead) .* exp(1i*2*pi*mean(internal_data.B0(:,:,:,:,cfg.scout_rep_sel),4) ...
                .* internal_data.TEs(nt_pick+Nt_ahead));
            clear temp_int
        end
        clear mi internal_data
        a_int_for_ratio = a_int_for_ratio .* reshape(mask_si,1,1,[]);

        ratios = abs(vec(a_int_for_ratio).') * conj(abs(vec(scout_for_ratio))) ...
            / (abs(vec(a_int_for_ratio).') * conj(abs(vec(a_int_for_ratio))));
        clear a_int_for_ratio
    else
        disp('only internal scout in use -- ratios = 1')
        ratios = 1;
    end
end

%% B0 estimation from scout; B0_scout_antiregi = B0_scout
temp = scout_img;
phase_diff_lr = angle(temp(:,:,:,3).*temp(:,:,:,5)./temp(:,:,:,4)./temp(:,:,:,4))/2;
phase_diff_lr(isnan(phase_diff_lr)) = 0;
temp = temp .* exp(-1i*phase_diff_lr .* reshape(((-1).^(0:size(temp,4)-1)+1)/2,1,1,1,[]));
df_pi = 450;
tmpl  = 1:min(10, Nt_scout);
[I, B0s, OR, ~] = Dictionaries_gen_B0(TEs_fid(tmpl), [-df_pi:df_pi]);
I_dic  = I;
Img    = exp(-1i*angle(temp(:,:,:,tmpl)));
Img    = reshape(Img, Ny_nav, Nz_nav, Nx_nav, []);
Img_norm = Img;
[B0_scout_antiregi, ~,~,~] = fit_B0_dic(Img_norm, Img, I_dic, I, B0s, OR, 18, useGPU);
clear temp Img Img_norm
%% B0 mask
disp('B0 mask')

cw = prctile(abs(vec(scout_img(:,:,:,1))), 99.9);

mask_B0_nav = squeeze(abs(scout_img(:,:,:,1,:))) > cw/10;

if ~cfg.no_nav
%% Anti-registration: navigator -> scout
a_nav_antiregi = zeros(Ny_nav,Nz_nav,Nx_nav,Nt_scout,nbins);
for ii_bin = 1:nbins
    [~,~,Th,~] = motion_ops([Ny_nav,Nz_nav,Nx_nav], motion_params_base(ii_bin,:)./[1 1 1 voxel_size], 0);
    a_nav_antiregi(:,:,:,:,ii_bin) = Th(a_nav_sense(:,:,:,:,ii_bin));
end

wind_x = hamming_flat(size_nav(2), ceil(size_nav(2)*1/3));
wind_y = hamming_flat(size_nav(1), ceil(size_nav(1)*1/3));
wind_z = hamming_flat(size_nav(3), ceil(size_nav(3)*1/3));
[X,Y,Z] = meshgrid(wind_x, wind_y, wind_z);
M_soft  = X.*Y.*Z;

B0_nav_antiregi = zeros(Ny_nav,Nz_nav,Nx_nav,nbins);
for ii_bin = 1:nbins
    if ii_bin ~= bin_center
        tmp = fft2c(fftc(B0_nav(:,:,:,ii_bin), 3));
        tmp = tmp .* M_soft;
        tmp = ifft2c(ifftc(tmp, 3));
        [~,~,Th] = im_rotate(size_nav, motion_params_base(ii_bin,:)./[1 1 1 voxel_size], tmp, 'linear');
        B0_nav_antiregi(:,:,:,ii_bin) = real(Th(tmp));
    else
        B0_nav_antiregi(:,:,:,ii_bin) = B0_nav(:,:,:,ii_bin);
    end
end
else
    % No nav data: nothing to anti-register against the scout. Per user
    % instruction, B0_nav_antiregi is just 0/[] in this case -- the joint
    % B0 decomposition that consumes it (motion_newton_round.m, st_flag==1)
    % is itself skipped whenever nav_rep<2, so this is never dereferenced.
    disp('no nav data -- B0_nav_antiregi = []')
    B0_nav_antiregi = [];
end

%% Optimization setup
SEs_perm = permute(SEs_nav, [2 3 1 4]);  % Ny_nav,Nz_nav,Nx_nav,Ncoils
% load([cfg.fminunc_path,filesep, 'fminunc_options']);
% options.MaxIterations = 1000;


%% a_nav_refs
a_nav_refs = reshape(scout_img, Ny_nav,Nz_nav,Nx_nav, 1, []);

%% generate Basis
Bas  = createSkopeBasis(size_nav, [4 4 4], 1:9);
Bas  = reshape(Bas, prod(size_nav), []);

%% ratios all
nshot_total = floor(shot * Nrep/shot_fid)*shot_fid;
N_perg = nshot_total/nav_rep

order_tmp = 1:nshot_total;
order_grp_tmp = ceil(order_tmp/N_perg);

ratios_all = ratios(order_grp_tmp);
if nbins > 1
    % Only meaningful with >=2 real bins (ratios(2) must exist) -- with
    % no nav data, nbins==1 and order_grp_tmp is 1 everywhere already, so
    % ratios_all is already the single scalar ratio broadcast to every
    % shot; this correction has nothing left to do.
    tmpl = find(order_tmp>8*shot_fid_in);
    tmps = find(order_grp_tmp<2);
    tmpl = intersect(tmpl,tmps);
    ratios_all(tmpl) = ratios(2);
    clear tmpl tmps
end
clear order*tmp

%% Populate output struct
out.voxel_size              = voxel_size;
out.size_nav                = size_nav;
out.nav_rep                 = nav_rep;
out.coil_sel                = coil_sel;
out.flip_flag               = flip_flag;
out.file_name               = file_name;
out.ky_fid                  = ky_fid;
out.kz_fid                  = kz_fid;
out.TEs_fid                 = TEs_fid;
out.shot_fid                = shot_fid;
out.shot_fid_in             = shot_fid_in;
out.shot_fid_gap            = shot_fid_gap;
out.shot                    = shot;
out.Nrep                    = Nrep;
out.ScanType                = pos_data.ScanType;                 

out.motion_params_base      = motion_params_base;
out.B0_nav_antiregi         = B0_nav_antiregi;
out.B0_scout_antiregi       = B0_scout_antiregi;
out.mask_B0_nav             = mask_B0_nav;
out.mask_si                 = mask_si;
out.ratios                  = ratios;
out.a_nav_refs              = a_nav_refs;
out.a_nav_sense             = a_nav_sense;
out.SEs_perm                = SEs_perm;
out.Bas                     = Bas;

out.nshot_total             = nshot_total;
out.ratios_all              = ratios_all;
out.nav_int                 = nav_int;
% pos_data handle (for Python export sections that read ky_fid, nav_int, etc.)

out.kspace_cart_orig_fid = kspace_cart_orig_fid;


end



