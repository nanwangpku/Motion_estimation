%% setup path
if isunix
    sep = '/';
else
    sep = '\';
end


% Yannick's motion operator
addpath(genpath('./matlab'))

%% set up function

vec = @(x)x(:);

%% load data
file_name = 'example_smena_data_meGRE2';
load(['.',filesep,'data',filesep,file_name])  % SMENA data
load('./data/fminunc_options')            % quasi newton option


%% data loading


kspace_cart_orig_fid = data_str.kspace_cart_orig_fid;  % this is the SMENA-nav data
a_nav_refs           = data_str.a_nav_refs;   % scout images
Bas                  = data_str.Bas;          % spherical harmonic basis
SEs_perm             = data_str.SEs_perm;     % sensitivity map
TEs_fid              = data_str.TEs_fid;      % TE for SMENA-nav
flip_flag            = data_str.flip_flag;    % ignord
size_nav             = data_str.size_nav;     % scout size
ky_fid               = data_str.ky_fid;       % SMENA ky
kz_fid               = data_str.kz_fid;       % SMENA kz

shot_fid             = data_str.shot_fid;     % SMENA shot
shot_fid_in          = data_str.shot_fid_in;  % same, SMENA shot, this will be different for st-SMENA
shot_fid_gap         = data_str.shot_fid_gap; % control temporal resolution of fitting


nshot_total          = data_str.nshot_total;  % number of shot for EPTI data/ multi-echo GRE data
ratios_all           = data_str.ratios_all;   % amplitude scale for early shots when not steady state

voxel_size           = data_str.voxel_size;   % the voxel size for SMENA; true data voxel is [1 1 1]


tmp = num2cell(size_nav);
[Ny_nav, Nz_nav, Nx_nav] = tmp{:};
clear tmp
kx_sel = ceil(Nx_nav/5+1):ceil(Nx_nav/5*4);
Ncoils = size(SEs_perm,4);
Bas_norm = Bas.*[1 10 10 10 100 100 100 100 100];


%% build up vectors for how many SMENA-nav are used for fitting. By default, everything

fid_idx = 1:nshot_total;


fid_idx = mod(fid_idx-1,shot_fid)+1; % this defines each fid's trajectory idx

fid_idx_in = floor((fid_idx-1)/shot_fid_in)+1;

fid_sel = 4; % every 4 shot of SMENA-nav is used for fitting
fid_dis = 20; % discard every a few shots after selection
fid_int = repmat([ones(1,fid_sel),zeros(1,fid_dis)],[1 ceil(nshot_total/(fid_sel+fid_dis))]);
fid_int = logical(fid_int(1:nshot_total));

%%

nt_sel = [1:6]


nt_sel_dB0 = [1 5]  % TE used for B0 fitting
nt_sel_mo  = [1 3]  % TE used for motion fitting
fid_int = repmat([ones(1,fid_sel),zeros(1,fid_dis)],[1 ceil(nshot_total/(fid_sel+fid_dis))]);
fid_int = logical(fid_int(1:nshot_total));

resp_bin_sep = floor(((1:nshot_total)-1)/ (fid_sel + fid_dis) )+1; % this helps map from fid group to all fid trajectory
nbins_sep = max(resp_bin_sep)

%% %%%%%%%%%%%%%%%%%%%%%
%% option 1: matlab quasi-newton fitting, this is slow
%% motion and dB0 estimation from real kspace
Nt_dB0 = numel(nt_sel_dB0);

Nt_mot = numel(nt_sel_mo);


kx_sel = [ceil(Nx_nav/5+1):ceil(Nx_nav/5*4)];


mask_dB0 = zeros(Ny_nav,Nz_nav,Nx_nav,1,Nt_dB0);

for it = 1:Nt_mot
    mask_dB0(ky_fid((1:shot_fid_in),nt_sel_dB0(it)),kz_fid((1:shot_fid_in),nt_sel_dB0(it)),kx_sel,1,it) = 1;
end

mask_mot = zeros(Ny_nav,Nz_nav,Nx_nav,1,Nt_mot);


for it = 1:Nt_mot
    mask_mot(ky_fid((1:shot_fid_in),nt_sel_mo(it)),kz_fid((1:shot_fid_in),nt_sel_mo(it)),kx_sel,1,it) = 1;
end



if flip_flag
    mask_dB0 = flip(flip(mask_dB0,1),2);
    mask_mot = flip(flip(mask_mot,1),2);
end


coil_sel = [1:Ncoils];

%%
nMot = 1;
nph = 1;
nB0 = 1;
dphs = ones(Ny_nav,Nz_nav,Nx_nav);
for ii_bin = 1:30:nbins_sep
    x_m0 = zeros(1,6);
    x_p0 = zeros(1,9);  % second order for background phase
    x_c0 = zeros(1,9);  % second order for B0
    nMot_total = 0;
    nB0_total = 0;
    nph_total = 0;
    ii_order = ceil(find(resp_bin_sep==ii_bin,1,'first')/shot_fid)


    for ii_iter = 1
        

        for ii_iter_B0 = 1:nB0

            nB0_total = nB0_total+1;
            display(['B0 : ', num2str(ii_bin),'/',num2str(nbins_sep),' : iter ',num2str(nB0_total)]);
            % t = [1 3]; % time point for B0 estimation. motion time points
            kspace_data = zeros(Ny_nav,Nz_nav,Nx_nav,Ncoils,numel(nt_sel_dB0));
            for ii = 1:4
                for it = 1:numel(nt_sel_dB0)
                    kspace_data(ky_fid(ii,nt_sel_dB0(it)),kz_fid(ii,nt_sel_dB0(it)),:,:,it) = fftc(permute(kspace_cart_orig_fid{ii_order}{ii}(:,nt_sel_dB0(it),:),[1 3 2]),1).*ratios_all(ii_bin*shot_fid); % Nx,Ncoils, nt_sel
                end
            end
            if flip_flag
                kspace_data = flip(flip(kspace_data,1),2);
            end
            in_str.size = size_nav;
            in_str.Bas = Bas_norm;
            in_str.mot = x_m0;
            in_str.bcoef = zeros(1,9);
            in_str.TEs = TEs_fid(nt_sel_dB0);  % input structure, TEs, in second
            in_str.pcoef = x_p0;
            in_str.dphs = dphs;      % background phs difference
            tic
            [c_params, loss_org] = fminunc(@(x)computeSensePlusdB0DataNew( x, a_nav_refs(:,:,:,1,nt_sel_dB0), in_str, kspace_data(:,:,:,coil_sel,:), SEs_perm(:,:,:,coil_sel), mask_dB0(:,:,:,1,1:numel(nt_sel_dB0))), x_c0, options);
            %                 c_params = x_c0;
            toc

            x_c0 = c_params;
            
        end
        % motion
        for ii_iter_mo = 1:nMot
            nMot_total = nMot_total+1;
            display(['motion : ', num2str(ii_bin),'/',num2str(nbins_sep),' : iter ',num2str(nMot_total)]);
            %             t = [2 4 6]
            %             t = [1 2]; % time point for B0 estimation. motion time points
            kspace_data = zeros(Ny_nav,Nz_nav,Nx_nav,Ncoils,numel(nt_sel_mo));
            for ii = 1:4
                for it = 1:numel(nt_sel_mo)
                    kspace_data(ky_fid(ii,nt_sel_mo(it)),kz_fid(ii,nt_sel_mo(it)),:,:,it) = fftc(permute(kspace_cart_orig_fid{ii_order}{ii}(:,nt_sel_mo(it),:),[1 3 2]),1).*ratios_all(ii_bin*shot_fid); % Nx,Ncoils, nt_sel
                end
            end
            if flip_flag
                kspace_data = flip(flip(kspace_data,1),2);
            end
            in_str.size = size_nav;  % input structure, image size
            in_str.Bas = Bas;        % input structure, basis for spherical homarnics, N (Nx*Ny*Nz) * Ncoef (9)
            in_str.mot = zeros(1,6); % input structure, motion params. if search for motion ,then set to be 0
            in_str.bcoef = x_c0;     % input structure, coef for spherical homarnics. if search for coef ,then set to be 0
            in_str.pcoef = x_p0;
            in_str.TEs = TEs_fid(nt_sel_mo);  % input structure, TEs, in second
            in_str.dphs = dphs;      % background phs difference
            tic
            [mot_params, loss_org] = fminunc(@(x)computeSensePlusMotionDataNew( x, a_nav_refs(:,:,:,1,nt_sel_mo),in_str,  kspace_data(:,:,:,coil_sel,:), SEs_perm(:,:,:,coil_sel), mask_mot(:,:,:,1,1:numel(nt_sel_mo))), x_m0, options);
            toc


            x_m0 = mot_params;

        end

        clear kspace_tmp


    end

    motion_est(ii_bin,:) = x_m0;
    cr_est(ii_bin,:) = x_c0;

end

%% save results
% saved here (not next to the input data) so recon_megre.m can load a single,
% predictable filename regardless of which estimation script produced it
dataset_path = pwd;
save([dataset_path,filesep,'data',filesep,'motion_estimation.mat'], ...
    'motion_est','cr_est','Bas','Bas_norm','voxel_size','size_nav','fid_sel','fid_dis','nshot_total')

