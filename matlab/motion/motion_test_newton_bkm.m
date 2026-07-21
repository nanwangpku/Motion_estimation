%% setup path
if isunix
    sep = '/';
else
    sep = '\';
end
main_path = '/local_mount/space/mayday/data/users/Nan/code';
bart_path = '/usr/local/app/bart/bart-0.6.00';
addpath(genpath(bart_path))
setenv('TOOLBOX_PATH',bart_path);

% Yannick's motion operator
addpath(genpath('/local_mount/space/mayday/data/users/yannick/Projects/MoB0-navigator'))
addpath(genpath('/local_mount/space/mayday/data/users/yannick/Software/Registration/rigidReg'))
addpath(genpath('/local_mount/space/mayday/data/users/yannick/Projects/Skope/Methods'));
addpath(genpath('/local_mount/space/mayday/data/users/yannick/Projects/Recon'));
addpath(genpath('/local_mount/space/mayday/data/users/yannick/Software/Utilities'));
addpath(genpath('/local_mount/space/mayday/data/users/yannick/Software/DISORDER'));

addpath(genpath('/local_mount/space/mayday/data/users/Nan/code/Funcs'))
% addpath(genpath('/local_mount/space/mayday/data/users/Nan/code/sLORAKS'))
addpath(genpath('/local_mount/space/mayday/data/users/Nan/code/motion'))
% addpath(fullfile(bart_path, 'matlab'));
% setenv('TOOLBOX_PATH',bart_path);

addpath(genpath(strcat(main_path, sep, 'cirEPTI')))
% addpath(fullfile(bart_path, 'matlab'));

%% 
% in this code, try to register everything to scout
%% use GPU
% useGPU=(exist('gpuNUFFT','file')>1); %true if using gpuNUFFT. false if using irt.
useGPU=false;
if useGPU
    try
        choose_GPU; %Give MATLAB main software access to the GPU before the gpuNUFFT software locks it up
    catch
        gpudev = gpuDevice(3);
    end
end
%% set up function

vec = @(x)x(:);


%% data loading
% dataset_path = '/local_mount/space/mayday/data/users/Nan/data_GE/LPCH/20260529_14454/';
% 
% 
% pos_path =     '11c31_crm17_40_54_long/recon/recon_11c31_data_sigpy_recon.mat'
% 
% pos_nav_path = '11c31_crm17_40_54_long/recon/recon_11c31_nav_sigpy_recon.mat'
% 
% pos_fid_path = '11c31_crm17_40_54_long/recon/recon_11c31_fid_sigpy.mat'
% 
% 
% scout_path = ['11c31_crm17_40_54_long/recon/recon_11c31_scout_sigpy_recon.mat'];
% 
% 
% 
% full_path = '10c32_full_low/recon/fits_60_NC48.mat'
data_path = '/local_mount/space/mayday/data/users/Nan/data_GE/'
dataset_path = uigetdir(data_path, 'which dataset');
dataset_path = [dataset_path,filesep];

dataseries_path = uigetdir(dataset_path, 'select the data series path to work on');

full_path = uigetdir(dataset_path, 'select the calibration path');
tmp_path = [full_path,filesep,'recon',filesep,'fits_60*'];
tmp_path = dir(tmp_path);
full_path = [tmp_path.folder, filesep, tmp_path.name];
clear tmp_path

pos_path = [dataseries_path,filesep,'recon',filesep,'recon_*data_sigpy_recon.mat'];
tmp_path = dir(pos_path)

if isempty(tmp_path) % old way of saving data
    pos_path = [dataseries_path,filesep,'recon',filesep,'recon_*_recon.mat'];
    tmp_path = dir(pos_path);
    pos_path = [tmp_path.folder, filesep, tmp_path.name];

    pos_nav_path = [];

    pos_fid_path = [dataseries_path,filesep,'recon',filesep,'recon_*_sigpy.mat'];
    tmp_path = dir(pos_fid_path);
    pos_fid_path = [tmp_path.folder, filesep, tmp_path.name];
    
    % in old way, scout alwasy separate
    scout_path = uigetdir(dataset_path, 'select scout path');
    tmp_path = dir([scout_path,filesep,'recon',filesep,'recon_*_recon.mat'])
    scout_path = [tmp_path.folder,filesep,tmp_path.name];

else  % new way of saving data
   
    pos_path = [tmp_path.folder, filesep, tmp_path.name];

    nav_path = [dataseries_path,filesep,'recon',filesep,'recon_*nav_sigpy_recon.mat'];
    tmp_path = dir(nav_path);
    pos_nav_path = [tmp_path.folder, filesep, tmp_path.name];

    fid_path = [dataseries_path,filesep,'recon',filesep,'recon_*fid_sigpy.mat'];
    tmp_path = dir(fid_path);
    pos_fid_path = [tmp_path.folder, filesep, tmp_path.name];


    scout_path = [dataseries_path,filesep,'recon',filesep,'recon_*scout_sigpy_recon.mat'];
    tmp_path = dir(scout_path);

    if isempty(tmp_path) % scout in separate
        scout_path = uigetdir(dataset_path, 'select scout path');
        tmp_path = dir([scout_path,filesep,'recon',filesep,'recon_*_recon.mat'])
        scout_path = [tmp_path.folder,filesep,tmp_path.name];

    else % scout in same folder
        scout_path = [tmp_path.folder, filesep, tmp_path.name];
    end
end

scout_path = erase(scout_path, dataset_path)
pos_path   = erase(pos_path, dataset_path)
pos_fid_path   = erase(pos_fid_path, dataset_path)
full_path  = erase(full_path, dataset_path)
try
pos_nav_path   = erase(pos_nav_path, dataset_path)
end

%
if contains(scout_path,'bkfs')
    scout_flag = 'bkfs';
elseif contains(scout_path,'bk_10_27')
    scout_flag = 'bk27';
else
    scout_flag = 'bk';
end

[tmp_path,~,~] = fileparts(pos_path);
file_path = [dataset_path, tmp_path, filesep];
clear tmp_path

%%

cfg.dataset_path    = dataset_path;
cfg.pos_path     = pos_path;
cfg.pos_nav_path = pos_nav_path;
cfg.pos_fid_path = pos_fid_path;
cfg.scout_path   = scout_path;
cfg.full_path    = full_path;
cfg.fminunc_path    = [main_path,filesep,'motion'];


if contains(pos_path,'fsm17')
    cfg.Nt_ahead    = 0;
    cfg.Nt_scout_in = [1:6 27:32 55:60];
else
    cfg.Nt_ahead    = 7;
    cfg.Nt_scout_in = 6;
end
cfg.bin_center      = 1;
cfg.nt_pick         = 3;
cfg.scout_rep_sel   = 2;
cfg.shot_fid_gap    = 1;


out = motion_newton_core(cfg);

%% Unpack for Python export
dataset_path         = cfg.dataset_path;
file_name            = out.file_name;

kspace_cart_orig_fid = out.kspace_cart_orig_fid;
ratios               = out.ratios;
a_nav_refs           = out.a_nav_refs;
Bas                  = out.Bas;
SEs_perm             = out.SEs_perm;
TEs_fid              = out.TEs_fid;
flip_flag            = out.flip_flag;
size_nav             = out.size_nav;
ky_fid               = out.ky_fid;
kz_fid               = out.kz_fid;

shot                 = out.shot;
shot_fid             = out.shot_fid;
shot_fid_in          = out.shot_fid_in;
shot_fid_gap         = out.shot_fid_gap;
nav_rep              = out.nav_rep;
Nrep                 = out.Nrep;

nshot_total          = out.nshot_total;
ratios_all           = out.ratios_all;

B0_nav_antiregi      = out.B0_nav_antiregi;
B0_scout_antiregi    = out.B0_scout_antiregi;
motion_params_base   = out.motion_params_base;
voxel_size           = out.voxel_size;
mask_B0_nav          = out.mask_B0_nav;
nav_int              = out.nav_int;

tmp = num2cell(size_nav);
[Ny_nav, Nz_nav, Nx_nav] = tmp{:};
clear tmp
kx_sel = ceil(Nx_nav/5+1):ceil(Nx_nav/5*4);
Ncoils = size(SEs_perm,4);



%% estimation parameters
if strcmp(out.ScanType, 'fs')
    fid_idx = 1:numel(out.nav_int);
    fid_idx = fid_idx(out.nav_int==1);

else

    fid_idx = 1:nshot_total;
end

fid_idx = mod(fid_idx-1,shot_fid)+1; % this defines each fid's trajectory idx

fid_idx_in = floor((fid_idx-1)/shot_fid_in)+1; 

fid_sel = 4; % select every a few shots for a separate estimation, better to be a integer multiple of shot_fid_in
fid_dis = 0; % discard every a few shots after selection
fid_int = repmat([ones(1,fid_sel),zeros(1,fid_dis)],[1 ceil(nshot_total/(fid_sel+fid_dis))]);
fid_int = logical(fid_int(1:nshot_total));

if strcmp(out.ScanType, 'fs')
    nt_sel_dB0 = [1 7]
    nt_sel_mo  = [1 3]
else
    nt_sel_dB0 = [1 5]
    nt_sel_mo  = [1 3]
end

resp_bin = floor(((1:nshot_total)-1)/ (fid_sel + fid_dis) )+1; % this helps map from fid group to all fid trajectory


%% prepare for estimation with gradient, thisestimation gives an answer every shot_fid_in * shot_fid_gap
motion_str = out;


timing_str.fid_idx    = fid_idx;
timing_str.fid_idx_in = fid_idx_in;
timing_str.resp_bin   = resp_bin;
timing_str.fid_int    = fid_int;  % which shot to keep or to discard
timing_str.nt_sel_dB0 = nt_sel_dB0;  % dB0 est
timing_str.nt_sel_mo  = nt_sel_mo;  % motion est

% init_str.B0_est_spatial = B0_est_poly;
init_str = struct();


est_out = motion_grad_est_spatial(motion_str, timing_str,init_str);

% nt_sel_dB0 = est_out.nt_sel_dB0;  % dB0 est
% nt_sel     = est_out.nt_sel;  % motion est

% save
m_est = est_out.m_est;
cr_est_poly = est_out.cr_est_poly;
c_est = est_out.c_est;
save([dataset_path,'estimation_',file_name,'_',scout_flag,'_sel',num2str(fid_sel),'_grad.mat'],'m_est','c_est','cr_est_poly','Bas','motion_params_base','B0_nav_antiregi','B0_scout_antiregi','voxel_size','mask_B0_nav','ratios','nt_sel*')


%% interp for better smoothes motion
motion_est_intp2 = medfilt1(m_est,lcm(shot_fid,fid_sel)/fid_sel);
% motion_est_all = reshape(repmat(reshape(motion_est_intp2,1,numel(fid_idx_in),6),[shot_fid_in 1 1]),[],6);
motion_est_all = interp1([1:(fid_sel+fid_dis):numel(fid_idx)], motion_est_intp2,[1:numel(fid_idx)],'linear','extrap');
%% B0 decomposition over multiple states, using B0_nav_antiregi

B0_var_gt = B0_nav_antiregi(:,:,:,[2:nav_rep])-mean(B0_nav_antiregi(:,:,:,2:nav_rep),4);

motion_coef_nav = cat(2,ones(nav_rep-1,1),motion_params_base(2:nav_rep,2:3));
nbasis = size(motion_coef_nav,2)
motion_weight_nav = motion_coef_nav'*motion_coef_nav
motion_coef_w_nav = real(motion_coef_nav/sqrtm(motion_weight_nav));
B0_coef_nav = reshape(B0_var_gt,[],size(motion_coef_w_nav,1))*motion_coef_w_nav;
B0_coef_nav = reshape(B0_coef_nav,[size_nav nbasis]);
B0_proj_nav = reshape(reshape(B0_coef_nav,[],nbasis)*motion_coef_w_nav',[size_nav size(motion_coef_nav,1)]);
% B0_proj_nav is the projected B0 using vNav

%% find those time slot where motion changes small, this is to generate a separated states for B0 proj estimation

diff1s = circshift(motion_est_all,[-1 0])-motion_est_all;
for i = 1:6
    diff1s(abs(diff1s(:,i))<0.1/fid_sel,i) = 0;
end
diff1s = sum(abs(diff1s),2);
diff1s(diff1s>0) = 1;
diff2s = diff1s - circshift(diff1s,[1 0]);
l_edm = find(diff2s==1);
size(l_edm)
l_bgm = find(diff2s==-1);
size(l_bgm)

% second round, remove the ones with monopolar changes, only keep the ones
% that are oscillate
tmp = zeros(numel(fid_idx_in),1);
for ii = 1:numel(l_edm)
    tmp (l_bgm(ii):l_edm(ii)) = 1;
    tmps = l_bgm(ii):l_edm(ii);
    mo_tmp = motion_est_all(tmps,:);
    mo_tmp = mo_tmp-mean(mo_tmp,1);
    for ii_mot = 1:6
        tmp(find(abs(mo_tmp(:,ii_mot))>1.2/fid_sel)+l_bgm(ii)-1) = 0;
    end
end
tmp = bwareaopen(tmp,4);  % remove all "1 block" than contains <=3 "1"s

% thrid round, update l_edm and l_bgm
diffs = [0;tmp;0;] - circshift([0;tmp;0],[-1 0]);
l_edm = find(diffs==1)-1;
size(l_edm)
l_bgm = find(diffs==-1);
size(l_bgm)
% [l_bgm,l_edm, cluster_id, labels_smooth, seg_table] = motion_temporal_cluster(motion_est_all);

%% prepare B0 basis and SEs for python
Bas_tmp = reshape(Bas,Ny_nav,Nz_nav,Nx_nav,9);
Bas_tmp = permute(Bas_tmp,[4 1 2 3]);

SEs_tmp = permute(SEs_perm,[4 1 2 3]);


%% 20260409, save to python, for joint navi, proj, a test case with selected data
data_label_test = 'joint_test';
mkdir([dataset_path,file_name,filesep,data_label_test,filesep])

if strcmp(out.ScanType, 'fs')
    nt_sel = [1:4 11:12]
else
    nt_sel = [1:6]
end
ky_sel = ky_fid(:,nt_sel);
kz_sel = kz_fid(:,nt_sel);
kx_sel = [ceil(Nx_nav/5+1):ceil(Nx_nav/5*4)];


shot_sel_test = [];
nbins_test = numel(l_edm);
for ii = 1:nbins_test % only test for first 5
    shot_sel_test = cat(2,shot_sel_test,l_bgm(ii)+1:l_edm(ii));
end
resp_bin_point = [0 l_edm(1:nbins_test).'];
% % for discrete resp_bin, following time only
resp_bin_test = zeros(1,resp_bin_point(end));

nbins_test = numel(resp_bin_point)-1
for i = 1:nbins_test
    resp_bin_test(resp_bin_point(i)+1:resp_bin_point(i+1)) = i;
end
resp_bin_test = resp_bin_test(shot_sel_test);

ratios_test = ratios_all(shot_sel_test);

if flip_flag
    ky_tmp = Ny_nav+1-ky_sel;
    kz_tmp = Nz_nav+1-kz_sel;
else
    ky_tmp = ky_sel;
    kz_tmp = kz_sel;
end

writecfl([dataset_path,file_name,filesep,data_label_test,filesep,'ky_tmp'],ky_tmp);
writecfl([dataset_path,file_name,filesep,data_label_test,filesep,'kz_tmp'],kz_tmp);

crds = zeros(numel(nt_sel),nshot_total,numel(kx_sel),3);
kspace_tmp = zeros(numel(nt_sel),Ncoils,nshot_total,numel(kx_sel));


for ii_all = 1:nshot_total
   
    ii_out = floor((ii_all-1)/shot_fid)+1;
    ii_in  = mod(ii_all-1,shot_fid)+1;
    tmp_sig = fftc(kspace_cart_orig_fid{ii_out}{ii_in}(:,nt_sel,:),1);
    kspace_tmp(:,:,ii_all,:) = permute(tmp_sig(kx_sel,:,:),[2 3 1]);
    % the crds is designed for python gridded nufft, with [0 N-1]
    crds(:,ii_all,:,3) = repmat((kx_sel-1),[numel(nt_sel) 1]);
    crds(:,ii_all,:,1) = repmat(ky_tmp(fid_idx(ii_all),:).',[1 numel(kx_sel)])-1;
    crds(:,ii_all,:,2) = repmat(kz_tmp(fid_idx(ii_all),:).',[1 numel(kx_sel)])-1;

end

crds = crds(:,shot_sel_test,:,:);
kspace_tmp = kspace_tmp(:,:,shot_sel_test,:);
clear tmp_sig

writecfl([dataset_path,file_name,filesep,data_label_test,filesep,'crds_tmp'],crds);

writecfl([dataset_path,file_name,filesep,data_label_test,filesep,'kspace_all_new_tmp'],kspace_tmp);
writecfl([dataset_path,file_name,filesep,data_label_test,filesep,'ratios_new_tmp'],ratios_test)
writecfl([dataset_path,file_name,filesep,data_label_test,filesep,'motion_bin_tmp'],resp_bin_test-1) % -1 is prepared for python
a0_tmp = permute(a_nav_refs(:,:,:,1,nt_sel),[5 1 2 3 4]);
writecfl([dataset_path,file_name,filesep,data_label_test,filesep,'a0_tmp'],a0_tmp);

% 

TEs_tmp = TEs_fid(nt_sel)
writecfl([dataset_path,file_name,filesep,data_label_test,filesep,'TEs_tmp'],TEs_tmp);


% motion_params_gt = motion_params_base(gt_order_spa,:) ./ [1 1 1 voxel_size];
motion_params_test = zeros(nbins_test,6);
motion_est_init = motion_est_all(shot_sel_test,:);
for ii_bin = 1:nbins_test
    motion_params_test(ii_bin,:) = mean(motion_est_init(resp_bin_test==ii_bin,:),1);
end

writecfl([dataset_path,file_name,filesep,data_label_test,filesep,'motion_est_tmp'],motion_params_test);


motion_coef_test = motion_params_test(:,2:3);
motion_coef_test = cat(2,ones(nbins_test,1),motion_coef_test);
nbasis = 3
motion_weight_test = motion_coef_test'*motion_coef_test;
motion_coef_w_test = motion_coef_test/sqrtm(motion_weight_test);
B0_proj_nav_to_test = reshape(reshape(B0_coef_nav,[],nbasis)*(motion_coef_test/sqrtm(motion_weight_nav))',[size_nav nbins_test]);
B0_coef_test_init = reshape(reshape(B0_proj_nav_to_test,[],nbins_test)*motion_coef_w_test,[size_nav nbasis]);


writecfl([dataset_path,file_name,filesep,data_label_test,filesep,'B0_coef_tmp'],permute(B0_coef_test_init,[4 1 2 3]));
writecfl([dataset_path,file_name,filesep,data_label_test,filesep,'B0_var_tmp'],permute(B0_proj_nav_to_test,[4 1 2 3]));


writecfl([dataset_path,file_name,filesep,data_label_test,filesep,'Bas_tmp'],Bas_tmp);
writecfl([dataset_path,file_name,filesep,data_label_test,filesep,'SEs_tmp'],SEs_tmp);
%% estimate with motion_est_dyn_app
display('joint est')
whichdev = choose_GPU();
% 1. Define the path to your Miniforge environment's libstdc++
miniforge_lib = '/home/wangnx/miniforge3/envs/mr_recon/lib/libstdc++.so.6';
python_path = '/local_mount/space/mayday/data/users/Nan/code/epti_python/mr_recon-main/motion_est_b_coef_dyn_app.py'

% 2. Construct the command, prepending the LD_PRELOAD env variable
cmd = [ ...
    'LD_PRELOAD=', miniforge_lib, ' ', ...
    '/home/wangnx/miniforge3/envs/mr_recon/bin/python ', python_path, ...
    ' --path ', [dataset_path, file_name, filesep, data_label_test], ...
    ' --gpu ', num2str(whichdev-1) ...
];

% 3. Execute the system command
% [status, cmdout] = system(cmd);
system(cmd)

% % Check for errors
% if status ~= 0
%     error('Python script failed with error:\n%s', cmdout);
% end

%% rebuild new estimated dB0 from test
B0_coef_test_est = readcfl([dataset_path,file_name,filesep,data_label_test,filesep,'b0_app_Adam_lam1e-08_it1000']);
B0_coef_test_est = ipermute(B0_coef_test_est,[4 1 2 3]);

B0_proj_test_est = reshape(reshape(B0_coef_test_est,[],nbasis)*motion_coef_w_test',[size_nav nbins_test]);
save([dataset_path,'estimation_',file_name,'_',scout_flag,'_nt',num2str(nt_sel(1)),'_',num2str(nt_sel(end)),'_sel',num2str(fid_sel),'_',data_label_test,'.mat'],'B0_coef_test_est','motion_coef_test','motion_coef_w_test','motion_weight_test','nbasis','size_nav','resp_bin_test','shot_sel_test','nbins_test','l_edm','l_bgm')
%% save to python, state_by_state, new batching
data_label_sep = 'sepbatch';
dataset_sep_path = [dataset_path,file_name,filesep,data_label_sep,filesep];
mkdir(dataset_sep_path)

if strcmp(out.ScanType, 'fs')
    nt_sel = [1:4 11:12]
else
    nt_sel = [1:6]
end
nt_sel_dB0 = [1 5]
nt_sel_mo  = [1 3]
% fid_sel = 12; % select every a few shots for a separate estimation, better to be a integer multiple of shot_fid_in
% fid_dis = 0; % discard every a few shots after selection
fid_int = repmat([ones(1,fid_sel),zeros(1,fid_dis)],[1 ceil(nshot_total/(fid_sel+fid_dis))]);
fid_int = logical(fid_int(1:nshot_total));

resp_bin_sep = floor(((1:nshot_total)-1)/ (fid_sel + fid_dis) )+1; % this helps map from fid group to all fid trajectory
nbins_sep = max(resp_bin_sep)



Nt_sel = numel(nt_sel);
ky_sel = ky_fid(:,nt_sel);
kz_sel = kz_fid(:,nt_sel);


if flip_flag
    ky_sel = Ny_nav+1-ky_sel;
    kz_sel = Nz_nav+1-kz_sel;

end


kspace_tmp = zeros(nbins_sep,Ncoils,Nt_sel,fid_sel,Nx_nav);
ky_tmp = zeros(nbins_sep,fid_sel,Nt_sel);
kz_tmp = zeros(nbins_sep,fid_sel,Nt_sel);

ratios_sep = zeros(1,nbins_sep);
for ii_bin = 1:nbins_sep

    ii_shot = find(resp_bin_sep == ii_bin);
    shot_per_bin = numel(ii_shot);
    for ii = 1:shot_per_bin
        ii_all = ii_shot(ii);
        ii_out = floor((ii_all-1)/shot_fid)+1;
        ii_in  = mod(ii_all-1,shot_fid)+1;
    
        ratios_sep(ii_bin) = ratios_all(ii_all);
        kspace_tmp(ii_bin,:,:,ii,:) = permute(fftc(kspace_cart_orig_fid{ii_out}{ii_in}(:,nt_sel,:),1),[3 2 1]);
        
        ky_tmp(ii_bin,ii,:) = ky_sel(fid_idx(ii_all),:);
        kz_tmp(ii_bin,ii,:) = kz_sel(fid_idx(ii_all),:);
    
    end
end

writecfl([dataset_sep_path,'ky_tmp'],ky_tmp);
writecfl([dataset_sep_path,'kz_tmp'],kz_tmp);
writecfl([dataset_sep_path,'kx_tmp'],kx_sel);


writecfl([dataset_sep_path,'kspace_all_new_tmp'],kspace_tmp);

writecfl([dataset_sep_path,'ratios_new_tmp'],ratios_sep)
a0_tmp = permute(a_nav_refs(:,:,:,1,nt_sel),[5 1 2 3 4]);
writecfl([dataset_sep_path,'a0_tmp'],a0_tmp);


TEs_tmp = TEs_fid(nt_sel);
writecfl([dataset_sep_path,'TEs_tmp'],TEs_tmp);
writecfl([dataset_sep_path,'nt_sel_mo'],nt_sel_mo);
writecfl([dataset_sep_path,'nt_sel_dB0'],nt_sel_dB0);

clear motion_init
for ii_bin = 1:nbins_sep
%     motion_init(ii_bin,:) = mean(motion_est_all((fid_sel+fid_dis)*(ii_bin-1)+(1:fid_sel+fid_dis),:),1);
    motion_init(ii_bin,:) = mean(motion_est_all(resp_bin_sep == ii_bin,:),1);
end

motion_coef_init = motion_init(:,2:3);
motion_coef_init = cat(2,ones(nbins_sep,1),motion_coef_init);
B0_proj_init = reshape(reshape(B0_coef_test_est,[],nbasis)*(motion_coef_init/sqrtm(motion_weight_test))',[size_nav nbins_sep]);



writecfl([dataset_sep_path,'motion_init'],motion_init); % -1 is prepared for python
writecfl([dataset_sep_path,'traj_bin'],fid_idx_in);
% write B0 baseline from different motionstate, optional
writecfl([dataset_sep_path,'B0_base_tmp'],permute(B0_proj_init,[4 1 2 3])); % -1 is prepared for python

% this part is same for all different methods
% writecfl([dataset_sep_path,'coil_sel_tmp'],coil_sel);
writecfl([dataset_sep_path,'Bas_tmp'],Bas_tmp);

writecfl([dataset_sep_path,'SEs_tmp'],SEs_tmp);

Bas_norm = Bas.*[1 10 10 10 100 100 100 100 100];
Basn_tmp = Bas_tmp.*([1 10 10 10 100 100 100 100 100].') ;
writecfl([dataset_sep_path,'Basn_tmp'],Basn_tmp);
%%

display('sep new batch')
whichdev = choose_GPU();
% 1. Define the path to your Miniforge environment's libstdc++
miniforge_lib = '/home/wangnx/miniforge3/envs/mr_recon/lib/libstdc++.so.6';

% 2. Construct the command, prepending the LD_PRELOAD env variable
python_path = '/local_mount/space/mayday/data/users/Nan/code/epti_python/mr_recon-main/motion_est_lbfgs.py'

cmd = [ ...
    'LD_PRELOAD=', miniforge_lib, ' ', ...
    '/home/wangnx/miniforge3/envs/mr_recon/bin/python ', python_path, ...
    ' --path ', [dataset_path, file_name, filesep, data_label_sep], ...
    ' --gpu ', num2str(whichdev-1) ...
];

% 3. Execute the system command
system(cmd);

%%
% disp('read adam')
% motion_est = readcfl([dataset_sep_path,'motion_est_adam']);
% cr_est     = readcfl([dataset_sep_path,'cr_est_adam']);
% motion_est_wfilt = medfilt1(motion_est,shot_fid/(fid_sel+fid_dis));
% save([dataset_path,'estimation_',file_name,'_',scout_flag,'_nt',num2str(nt_sel(1)),'_',num2str(nt_sel(end)),'_adam.mat'],'motion_est','motion_est_wfilt','cr_est','Bas','motion_params_base*','B0_nav_antiregi','B0_scout_antiregi','voxel_size','mask_B0_nav','ratios','ratios_sep','a_nav_refs','TEs_fid','fid_idx','fid_sel','fid_dis','ky_tmp','kz_tmp','resp_bin_sep')
%%
disp('read quasi newton')
motion_est = readcfl([dataset_sep_path,'motion_est_new']);
cr_est     = readcfl([dataset_sep_path,'cr_est_new']);
motion_est_wfilt = medfilt1(motion_est,lcm(shot_fid,(fid_sel+fid_dis))/(fid_sel+fid_dis));
cr_est_wfilt = medfilt1(cr_est,lcm(shot_fid,(fid_sel+fid_dis))/(fid_sel+fid_dis));

save([dataset_path,'estimation_',file_name,'_',scout_flag,'_nt',num2str(nt_sel(1)),'_',num2str(nt_sel(end)),'_sel',num2str(fid_sel),'_qn.mat'],'motion_est','motion_est_wfilt','cr_est','cr_est_wfilt','Bas','motion_params_base*','B0_nav_antiregi','B0_scout_antiregi','voxel_size','mask_B0_nav','ratios','ratios_sep','a_nav_refs','TEs_fid','fid_idx','fid_sel','fid_dis','ky_tmp','kz_tmp','resp_bin_sep')

%% save for correction
motion_est1 = motion_est_wfilt .*[1 1 1 4 4 4];
B0_est_2nd = reshape(Bas_norm*(cr_est_wfilt.'),[size_nav size(cr_est_wfilt,1)]);
B0_est_2nd = single(B0_est_2nd);
save([file_path,'motion_params_',scout_flag,'_nt',num2str(nt_sel(1)),'_',num2str(nt_sel(end)),'_sel',num2str(fid_sel),'.mat'],'motion_est1','B0_est_2nd','B0_proj_init','resp_bin_sep')

%% joint quasi newton
motion_est_qnj = readcfl([dataset_sep_path,'motion_est_reg3']);
cr_est_qnj     = readcfl([dataset_sep_path,'cr_est_reg3']);
save([dataset_path,'estimation_',file_name,'_',scout_flag,'_nt',num2str(nt_sel(1)),'_',num2str(nt_sel(end)),'_sel',num2str(fid_sel),'_qnj.mat'],'motion_est','motion_est_wfilt','cr_est','Bas','motion_params_base*','B0_nav_antiregi','B0_scout_antiregi','voxel_size','mask_B0_nav','ratios','ratios_sep','a_nav_refs','TEs_fid','fid_idx','fid_sel','fid_dis','ky_tmp','kz_tmp','resp_bin_sep')
