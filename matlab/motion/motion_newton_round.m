function r = motion_newton_round(p, motion_est_intp2, basis_sel, recompute_basis_sel, suffix)
% One round of spatiotemporal B0 decomposition + per-shot motion/B0 estimation.
% motion_test_newton_bkm calls this once for the initial estimate, and, for
% optionally a second time using round 1's result as a warm start (suffix =
% '_2nd', recompute_basis_sel = false so the basis chosen in round 1 is reused).
%
% p.st_flag selects which spatiotemporal (st) mode(s) to run:
%   0      - motion-only estimation, via motion_est_lbfgs.py, no B0 init
%   1      - spatiotemporal B0 decomposition (B0_init), then motion
%            estimation via motion_est_lbfgs.py using that B0_init
%   'both' - run st = 0 first, then run st = 1 warm-started from the
%            st = 0 motion estimate; the st = 1 result is returned as the
%            main output, with the st = 0 result attached as r.round_st0
%
% Optional fields on p, all set from motion_test_newton_bkm.m's USER OPTIONS
% block (see there). Absent or empty means "behave exactly as before":
%   p.sepmo_python   full path to the per-shot solver script. Default is
%                    motion_est_lbfgs.py. motion_est_lbfgs_fast.py is a
%                    drop-in replacement with the same inputs/outputs.
%   p.sepmo_tag      solver label of the per-shot fit ('' or '_fast[_alt<N>]'),
%                    part of the sepmo/jointB0/motion_est names, so a run
%                    with a different solver cannot collide with, or be
%                    silently loaded from, an existing cache.
%   p.run_tag        free-form string (opts.tag) at the end of those names.
%   p.joint_python   full path to the joint-B0 script. Default is
%                    motion_est_b_coef_dyn_app.py.
%                    motion_est_b_coef_dyn_app_fast.py is a drop-in
%                    replacement with the same inputs/outputs.
%   p.joint_tag      solver label of the joint-B0 fit ('' or '_jfast'), in the
%                    jointB0 and st=1 sepmo/motion_est names.
%
% File names (absolute echo numbers run together, e.g. [1 4 5 6] -> 1456):
%   <n>sepmo_<base>_coil<nc>[_coilmo<n>]_st<0|1>_mo<e>_s<e><sepmo_tag>[<joint>]<run_tag><suffix>
%   <n>jointB0_<base>_coil<ncj>_mo<e>_s<e><sepmo_tag><joint><run_tag><suffix>
%   motion_est_<base>_coil<nc>...           (as sepmo, in <series>/recon/)
%   <joint> = _j<e><joint_tag>_lam<lam_joint>_basis<k>, st=1 only
% smooth_flag, rotcoil_flag, lam_joint, basis_sel and nt_dB0_joint are saved
% inside the st=1 and jointB0 files instead of in the name.
%   p.keep_joint_dir true keeps the jointB0 export directory at the end of
%                    the round instead of deleting it (default false).
%   p.export_tag     string appended to the temporary sepmo/ and jointB0/
%                    export folder names only (result names are unchanged),
%                    so runs on the same series can proceed side by side.
%   p.nt_sel_joint   echo indices for the spatiotemporal B0 decomposition
%   p.nt_sel_sep     echo indices written out for the per-shot fit
%   p.nt_sel_dB0_sep echo indices the per-shot fit uses for its B0 sub-fit
%   p.nt_sel_mo_sep  echo indices the per-shot fit uses for its motion sub-fit

st_mode = p.st_flag;
is_both = (ischar(st_mode) && strcmpi(st_mode,'both')) || ...
    (isnumeric(st_mode) && numel(st_mode)>1 && all(ismember([0 1],st_mode)));

if is_both
    %% 1. run st = 0 first: motion-only estimation via motion_est_lbfgs.py
    p0 = p;
    p0.st_flag = 0;
    r0 = motion_newton_round_single(p0, motion_est_intp2, basis_sel, recompute_basis_sel, suffix);

    %% 2. then run st = 1: estimate B0_init, then motion via motion_est_lbfgs.py
    %% with B0_init, warm-started from the st = 0 motion estimate
    p1 = p;
    p1.st_flag = 1;
    r1 = motion_newton_round_single(p1, r0.motion_est_wfilt, r0.basis_sel, false, suffix);

    if isempty(r1)
        % st=1 failed outright (too few motion-state bins for the joint B0
        % decomposition, or the joint B0 estimate itself diverged) -- fall
        % back to the st=0 result entirely rather than a half st=1 result
        % with no B0 background.
        warning('motion_newton_round:st1Failed', ...
            'st=1 failed for this round -- falling back to the st=0 result.');
        r = r0;
    else
        r = r1;
    end
    r.round_st0 = r0;
else
    p.st_flag = st_mode;
    r = motion_newton_round_single(p, motion_est_intp2, basis_sel, recompute_basis_sel, suffix);
end

end

function r = motion_newton_round_single(p, motion_est_intp2, basis_sel, recompute_basis_sel, suffix)

dataset_path         = p.dataset_path;
file_name            = p.file_name;
ScanType             = p.ScanType;
pos_path             = p.pos_path;
file_series          = p.file_series;
file_path            = p.file_path;
save_base            = p.save_base;
Ncoils               = p.Ncoils;
lam_joint            = p.lam_joint;
smooth_flag          = p.smooth_flag;
rotcoil_flag         = p.rotcoil_flag;
ky_fid               = p.ky_fid;
kz_fid               = p.kz_fid;
Nx_nav               = p.Nx_nav;
Ny_nav               = p.Ny_nav;
Nz_nav               = p.Nz_nav;
flip_flag            = p.flip_flag;
ratios_all           = p.ratios_all;
ratios               = p.ratios;
nshot_total          = p.nshot_total;
shot_fid             = p.shot_fid;
kspace_cart_orig_fid = p.kspace_cart_orig_fid;
fid_idx              = p.fid_idx;
fid_idx_in           = p.fid_idx_in;
a_nav_refs           = p.a_nav_refs;
TEs_fid              = p.TEs_fid;
Bas_tmp              = p.Bas_tmp;
SEs_tmp              = p.SEs_tmp;
Basn_tmp             = p.Basn_tmp;
B0_nav_antiregi      = p.B0_nav_antiregi;
B0_scout_antiregi    = p.B0_scout_antiregi;
nav_rep              = p.nav_rep;
motion_params_base   = p.motion_params_base;
size_nav             = p.size_nav;
st_flag              = p.st_flag;
fid_sel              = p.fid_sel;
fid_dis              = p.fid_dis;
Bas                  = p.Bas;
Bas_norm             = p.Bas_norm;
voxel_size           = p.voxel_size;
mask_B0_nav          = p.mask_B0_nav;

% Optional differentiated coil-compression variants (coil_sel_mode='split'
% in motion_test_newton_bkm.m). When absent, everything below falls back
% to the single shared SEs_tmp/kspace_cart_orig_fid/Ncoils, i.e. today's
% default behaviour is unchanged.
if isfield(p,'SEs_tmp_jointb0') && ~isempty(p.SEs_tmp_jointb0)
    SEs_tmp_jointb0              = p.SEs_tmp_jointb0;
    kspace_cart_orig_fid_jointb0 = p.kspace_cart_orig_fid_jointb0;
    Ncoils_jointb0               = p.Ncoils_jointb0;
else
    SEs_tmp_jointb0              = SEs_tmp;
    kspace_cart_orig_fid_jointb0 = kspace_cart_orig_fid;
    Ncoils_jointb0               = Ncoils;
end
has_mo_variant = isfield(p,'SEs_tmp_mo') && ~isempty(p.SEs_tmp_mo);
if has_mo_variant
    SEs_tmp_mo              = p.SEs_tmp_mo;
    kspace_cart_orig_fid_mo = p.kspace_cart_orig_fid_mo;
end

% Optional overrides (see the header). Empty/absent keeps the original
% sequence-dependent choices made further down.
sepmo_python    = p_get(p, 'sepmo_python', ...
    fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
             '..','python','motion_est_lbfgs_fast.py'));
sepmo_tag       = p_get(p, 'sepmo_tag',       '');
sepmo_n_alt     = p_get(p, 'sepmo_n_alt',     1);
joint_python    = p_get(p, 'joint_python', ...
    fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
             '..','python','motion_est_b_coef_dyn_app.py'));
joint_tag       = p_get(p, 'joint_tag',       '');
keep_joint_dir  = p_get(p, 'keep_joint_dir',  false);
export_tag      = p_get(p, 'export_tag',      '');
% the st=1 per-shot fit starts from the joint B0, so its names carry joint_tag
if st_flag
    st_joint_tag = joint_tag;
else
    st_joint_tag = '';
end
nt_sel_joint_in = p_get(p, 'nt_sel_joint',    []);
nt_dB0_joint_in = p_get(p, 'nt_sel_dB0_joint',[]);
nt_sel_sep_in   = p_get(p, 'nt_sel_sep',      []);
nt_dB0_sep_in   = p_get(p, 'nt_sel_dB0_sep',  []);
nt_mo_sep_in    = p_get(p, 'nt_sel_mo_sep',   []);
run_tag         = p_get(p, 'run_tag',         '');

% per-shot echoes (absolute), needed by the jointB0 name as well as below
[nt_mo_abs, nt_dB0_abs] = sep_echoes(ScanType, pos_path, nt_mo_sep_in, nt_dB0_sep_in);
sep_block = ['_mo',echo_label(nt_mo_abs),'_s',echo_label(nt_dB0_abs),sepmo_tag];
sep_is_default   = isempty(nt_mo_sep_in) && isempty(nt_dB0_sep_in) && isempty(nt_sel_sep_in);
joint_is_default = isempty(nt_dB0_joint_in) && isempty(nt_sel_joint_in);

kx_sel = ceil(Nx_nav/5+1):ceil(Nx_nav/5*4);

motion_est_all = interp1([1:(fid_sel+fid_dis):numel(fid_idx)], motion_est_intp2,[1:numel(fid_idx)],'linear','extrap');

%% spatialtemporal
st1_failed = false;
if st_flag == 1
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
    if recompute_basis_sel
        if numel(l_bgm)<=7 && numel(l_bgm)>1
            basis_sel_order = [2 3 5 6 1 4];
            basis_sel = basis_sel_order(1:numel(l_bgm)-1);
        
        end
    end
    %% B0 decomposition over multiple states, using B0_nav_antiregi
    % nav_rep<2 means no real navigator data was available (nav is not
    % critical to motion estimation -- see motion_newton_core.m's cfg.no_nav
    % path) -- there's no real per-bin nav variance to build a B0 basis
    % from. Still run the joint B0 fit below, just seed it with an all-zero
    % initial B0 guess: B0_coef_nav=0 with motion_weight_nav=eye carries
    % through the exact same projection formulas further down (B0_proj_nav_to_test,
    % B0_coef_test_init) and yields exactly zeros there too, with no separate
    % special-casing needed at those call sites.
    if nav_rep >= 2
        B0_var_gt = B0_nav_antiregi(:,:,:,[2:nav_rep])-mean(B0_nav_antiregi(:,:,:,2:nav_rep),4);
        motion_coef_nav = cat(2,ones(nav_rep-1,1),motion_params_base(2:nav_rep,basis_sel));
        nbasis = size(motion_coef_nav,2)  % numel(basis_sel)+1
        motion_weight_nav = motion_coef_nav'*motion_coef_nav
        motion_coef_w_nav = real(motion_coef_nav/sqrtm(motion_weight_nav));
        B0_coef_nav = reshape(B0_var_gt,[],size(motion_coef_w_nav,1))*motion_coef_w_nav;
        B0_coef_nav = reshape(B0_coef_nav,[size_nav nbasis]);
        B0_proj_nav = reshape(reshape(B0_coef_nav,[],nbasis)*motion_coef_w_nav',[size_nav size(motion_coef_nav,1)]);
        % B0_proj_nav is the projected B0 using vNav
    else
        disp('no nav data -- seeding joint B0 fit with a zero initial guess')
        nbasis = numel(basis_sel)+1;
        motion_weight_nav = eye(nbasis);
        B0_coef_nav = zeros([size_nav nbasis]);
    end

    %% 20260409, save to python, for joint navi, proj, a test case with selected data
    data_label_test = 'jointB0';
    dataset_joint_path = [dataset_path,file_name,filesep,data_label_test,export_tag,filesep];

    mkdir(dataset_joint_path)

    if strcmp(ScanType, 'fs') && (contains(pos_path,'fsm17') || contains(pos_path,'fsm16'))
        nt_sel = [1:5 7:9 ]
    else
        nt_sel = [1:6]
    end
    % the joint stage has a single fit, so its echo set IS its B0 echo set:
    % nt_sel is derived from nt_sel_dB0_joint; nt_sel_joint still honoured.
    nt_dB0 = nt_sel;
    if ~isempty(nt_dB0_joint_in), nt_dB0 = nt_dB0_joint_in(:).'; end
    if ~isempty(nt_sel_joint_in)
        nt_sel = nt_sel_joint_in(:).';
    else
        nt_sel = nt_dB0;
    end
    fprintf('[jointB0] echoes nt_sel = %s | B0 fit uses %s\n', ...
            mat2str(nt_sel), mat2str(nt_dB0));

    nt_dB0_joint = nt_dB0;
    joint_block = ['_j',echo_label(nt_dB0_joint),joint_tag,'_lam',num2str(lam_joint),'_basis',num2str(numel(basis_sel))];
    % the st=1 joint fit starts from the st=0 per-shot motion, so its name
    % carries that fit's block too
    save_joint = [file_series, data_label_test,'_',save_base,'_coil',num2str(Ncoils_jointb0),sep_block,joint_block,run_tag,suffix]
    % a cache under the previous naming is reused if it provably matches
    save_joint_legacy = [file_series, data_label_test,'_',save_base,'_coil',num2str(Ncoils_jointb0),'_nt',num2str(nt_sel(1)),'_',num2str(nt_sel(end)),'_lam',num2str(lam_joint),'_basis',num2str(numel(basis_sel)),'_sB',num2str(smooth_flag),rotcoil_flag,joint_tag,sepmo_tag,run_tag,suffix];
    if ~exist([dataset_path,save_joint,'.mat'],'file') && ...
            legacy_ok([dataset_path,save_joint_legacy,'.mat'], struct('nt_dB0',nt_dB0_joint), ...
                      joint_is_default, sep_is_default || ~isempty(run_tag))
        fprintf('[jointB0] reusing cache under its old name %s\n', save_joint_legacy);
        save_joint = save_joint_legacy;
    end

    ky_sel = ky_fid(:,nt_sel);
    kz_sel = kz_fid(:,nt_sel);

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

    min_nbins_joint = 3;
    if nbins_test < min_nbins_joint
        % Not enough distinct motion states in this dataset to constrain the
        % joint B0 decomposition -- the python joint estimation would
        % otherwise run on a degenerate/rank-deficient problem and either
        % error out or silently return NaN. Skip the ENTIRE st=1 round for
        % this round (not just the B0 background): see st1_failed handling
        % right after this if/else block.
        warning('motion_newton_round:notEnoughMotionStates', ...
            ['only %d motion-state bin(s) detected (need >=%d) -- not enough ' ...
             'motion in this dataset for joint B0 estimation; skipping the ' ...
             'entire st=1 round.'], nbins_test, min_nbins_joint);
        st1_failed = true;
    else

    ratios_test = ratios_all(shot_sel_test);

    if flip_flag
        ky_tmp = Ny_nav+1-ky_sel;
        kz_tmp = Nz_nav+1-kz_sel;
    else
        ky_tmp = ky_sel;
        kz_tmp = kz_sel;
    end

    writecfl([dataset_joint_path,'ky_tmp'],ky_tmp);
    writecfl([dataset_joint_path,'kz_tmp'],kz_tmp);

    crds = zeros(numel(nt_sel),nshot_total,numel(kx_sel),3);
    kspace_tmp = zeros(numel(nt_sel),Ncoils_jointb0,nshot_total,numel(kx_sel));

    for ii_all = 1:nshot_total
        ii_out = floor((ii_all-1)/shot_fid)+1;
        ii_in  = mod(ii_all-1,shot_fid)+1;
        tmp_sig = fftc(kspace_cart_orig_fid_jointb0{ii_out}{ii_in}(:,nt_sel,:),1);
        kspace_tmp(:,:,ii_all,:) = permute(tmp_sig(kx_sel,:,:),[2 3 1]);
        % the crds is designed for python gridded nufft, with [0 N-1]
        crds(:,ii_all,:,3) = repmat((kx_sel-1),[numel(nt_sel) 1]);
        crds(:,ii_all,:,1) = repmat(ky_tmp(fid_idx(ii_all),:).',[1 numel(kx_sel)])-1;
        crds(:,ii_all,:,2) = repmat(kz_tmp(fid_idx(ii_all),:).',[1 numel(kx_sel)])-1;
    end

    crds = crds(:,shot_sel_test,:,:);
    kspace_tmp = kspace_tmp(:,:,shot_sel_test,:);
    clear tmp_sig

    writecfl([dataset_joint_path,'crds_tmp'],crds);

    writecfl([dataset_joint_path,'kspace_all_new_tmp'],kspace_tmp);
    writecfl([dataset_joint_path,'ratios_new_tmp'],ratios_test)
    writecfl([dataset_joint_path,'motion_bin_tmp'],resp_bin_test-1) % -1 is prepared for python
    a0_tmp = permute(a_nav_refs(:,:,:,1,nt_sel),[5 1 2 3 4]);
    writecfl([dataset_joint_path,'a0_tmp'],a0_tmp);

    TEs_tmp = TEs_fid(nt_sel)
    writecfl([dataset_joint_path,'TEs_tmp'],TEs_tmp);

    % motion_params_gt = motion_params_base(gt_order_spa,:) ./ [1 1 1 voxel_size];
    motion_params_test = zeros(nbins_test,6);
    motion_est_init = motion_est_all(shot_sel_test,:);
    for ii_bin = 1:nbins_test
        motion_params_test(ii_bin,:) = mean(motion_est_init(resp_bin_test==ii_bin,:),1);
    end

    writecfl([dataset_joint_path,'motion_est_tmp'],motion_params_test);

    motion_coef_test = motion_params_test(:,basis_sel);
    %     motion_coef_test = motion_coef_test-mean(motion_coef_test,1);% center to 0
    motion_coef_test = cat(2,ones(nbins_test,1),motion_coef_test);
    nbasis = size(motion_coef_test,2);
    motion_weight_test = motion_coef_test'*motion_coef_test;
    motion_coef_w_test = motion_coef_test/sqrtm(motion_weight_test);
    B0_proj_nav_to_test = reshape(reshape(B0_coef_nav,[],nbasis)*(motion_coef_test/sqrtm(motion_weight_nav))',[size_nav nbins_test]);
    B0_coef_test_init = reshape(reshape(B0_proj_nav_to_test,[],nbins_test)*motion_coef_w_test,[size_nav nbasis]);

    writecfl([dataset_joint_path,'basis_sel'],basis_sel);
    writecfl([dataset_joint_path,'B0_coef_tmp'],permute(B0_coef_test_init,[4 1 2 3]));
    writecfl([dataset_joint_path,'B0_var_tmp'],permute(B0_proj_nav_to_test,[4 1 2 3]));

    writecfl([dataset_joint_path,'Bas_tmp'],Bas_tmp);
    writecfl([dataset_joint_path,'SEs_tmp'],SEs_tmp_jointb0);
    %% estimate with motion_est_dyn_app

    if ~exist([dataset_path,save_joint,'.mat'],'file')
        display('joint est')
        whichdev = choose_GPU();
        % 1. Define the path to your Miniforge environment's libstdc++
        pyenv_ = motion_python_env(); miniforge_lib = pyenv_.libstdcxx;
        python_path = joint_python

        % 2. Construct the command, prepending the LD_PRELOAD env variable
        % NW/edit 20260910: also force LD_LIBRARY_PATH to prefer the
        % mr_recon env's own pip-installed nvidia-cusolver-cu11/
        % nvidia-cublas-cu11 over whatever system-wide CUDA happens to be
        % on this node -- this system() call invokes the env's python
        % binary directly rather than via `conda activate`, so the env's
        % own etc/conda/activate.d hook (which sets this) never runs.
        % Without it, torch.linalg calls (e.g. torch.linalg.eigh) can fail
        % with "undefined symbol: cusolverDnXsytrs, version
        % libcusolver.so.11" on nodes whose system CUDA predates that
        % symbol.
        nvidia_lib = pyenv_.nvidia;

        if isempty(rotcoil_flag)
            rotcoil_arg = '';
        else
            rotcoil_arg = [' --', rotcoil_flag(2:end)];   % ' --rotcoil'
        end

        cmd = [ ...
            'LD_PRELOAD=', miniforge_lib, ' ', ...
            'LD_LIBRARY_PATH=', nvidia_lib, ' ', ...
            pyenv_.bin, ' ', python_path, ...
            ' --path ', dataset_joint_path, ...
            ' --gpu ', num2str(whichdev-1) ...
            ' --lambda_tv ',num2str(lam_joint) ...
            rotcoil_arg ...
            ];

        % 3. Execute the system command
        % motion_est_b_coef_dyn_app.py exits nonzero (and skips saving a
        % result) if the loss is non-finite on its very first iteration --
        % a degenerate problem that won't recover over more iterations. Only
        % read back a result when it actually reports success.
        [status, cmdout] = system(cmd, '-echo');

        if status ~= 0
            warning('motion_newton_round:jointB0Failed', ...
                ['joint B0 estimation script failed (status %d) -- treating ' ...
                 'st=1 as failed for this round.\n%s'], status, cmdout);
            st1_failed = true;
        else
            %% rebuild new estimated dB0 from test
            clear B0_coef_smooth
            B0_coef_test_est = readcfl([dataset_joint_path,'b0_app_Adam_lam',num2str(lam_joint),'_it1000',rotcoil_flag]);
            B0_coef_test_est = ipermute(B0_coef_test_est,[4 1 2 3]);

            for ii = 1:nbasis
                B0_coef_smooth(:,:,:,ii) = imgaussfilt3(B0_coef_test_est(:,:,:,ii),0.5);
            end

            if smooth_flag  % additional smooth with espirit
                dt = 3e-3
                a_nav_smooth = cat(4,abs(a_nav_refs(:,:,:,1)), abs(a_nav_refs(:,:,:,1)).*exp(1i*2*pi*B0_coef_smooth.*dt));
                tic
                SEs_phase = bart('ecalib -c 0 -m 1 -r 60:54:60 -k 9:9:9', fft3c(a_nav_smooth));
                toc
                B0_coef_smooth = angle(SEs_phase(:,:,:,2:end)./SEs_phase(:,:,:,1))/2/pi/dt;
            end
            save([dataset_path,save_joint],'B0_coef_test_est','motion_coef_test','motion_coef_w_test','motion_weight_test','nbasis','size_nav','resp_bin_test','shot_sel_test','nbins_test','l_edm','l_bgm','B0_coef_smooth','basis_sel','nt_sel','nt_dB0','nt_dB0_joint','smooth_flag','rotcoil_flag','lam_joint')
        end
    else  % already saved
        load([dataset_path,save_joint])
    end

    if ~st1_failed
        if any(isnan(B0_coef_test_est(:)))
            % Same fallback as the nbins_test<min_nbins_joint guard above,
            % for the case where there were nominally enough bins but the
            % joint estimation was still degenerate (e.g. bins too similar
            % to each other) and came back NaN past the first-iteration
            % check in the python script.
            warning('motion_newton_round:jointB0NaN', ...
                'joint B0 estimation returned NaN values -- treating st=1 as failed for this round.');
            st1_failed = true;
        else
            B0_proj_test_est = reshape(reshape(B0_coef_test_est,[],nbasis)*motion_coef_w_test',[size_nav nbins_test]);

            B0_proj_smooth = reshape(reshape(B0_coef_smooth,[],nbasis)*motion_coef_w_test',[size_nav nbins_test]);
        end
    end
    end
end

if st1_failed
    % Skip the entire st=1 round (not just the B0 background): no separate
    % per-shot motion_est_lbfgs.py pass either. Clean up the joint-B0 temp
    % dir before bailing -- dataset_sep_path was never created on this path.
    if exist('dataset_joint_path','var') && exist(dataset_joint_path,'dir') && ~keep_joint_dir
        rmdir(dataset_joint_path,'s');
    end
    r = [];
    return
end
%% save to python, state_by_state, new batching
data_label_sep = 'sepmo';
dataset_sep_path = [dataset_path,file_name,filesep,data_label_sep,export_tag,filesep];

mkdir(dataset_sep_path)

% --- echo selection: two index conventions meet here ---------------------
% nt_sel      ABSOLUTE echo numbers. Picks which echoes get exported:
%             kspace_cart_orig_fid(:,nt_sel,:), TEs_fid(nt_sel),
%             a_nav_refs(...,nt_sel), ky_fid(:,nt_sel).
% nt_sel_dB0  POSITIONS WITHIN nt_sel, not absolute echoes, because the
% nt_sel_mo   python side indexes arrays that are ALREADY restricted to
%             nt_sel (tes[nt_b0], img[nt_mo], ksp[:,nt_b0], ...).
%             The fs defaults above show it: nt_sel = [1:4 7:8] paired with
%             nt_sel_dB0 = [1 5] means echoes 1 and 7, not 1 and 5.
%
% The p.nt_sel_*_sep overrides are supplied as ABSOLUTE echo numbers -- the
% same convention as the gradient step -- and converted to positions here,
% so a caller never has to keep the two straight.
%
% nt_sel is DERIVED: it is just the union of the echoes the two sub-fits
% need, so a caller supplies nt_sel_mo_sep and nt_sel_dB0_sep only. Nothing
% downstream reads an exported echo that neither sub-fit uses (the solver
% indexes tes[nt_b0], tes[nt_mo], img[...] and the matching k-space rows),
% so exporting the union is equivalent to exporting more, and cheaper.
% p.nt_sel_sep is still honoured if given, for backward compatibility.

% nt_mo_abs / nt_dB0_abs (absolute) were resolved at the top, by sep_echoes

if ~isempty(nt_sel_sep_in)
    nt_sel = nt_sel_sep_in(:).';            % explicit override
else
    nt_sel = union(nt_mo_abs, nt_dB0_abs);  % sorted, unique
end
nt_sel_dB0 = nt_abs2pos(nt_dB0_abs, nt_sel, 'nt_sel_dB0_sep');
nt_sel_mo  = nt_abs2pos(nt_mo_abs,  nt_sel, 'nt_sel_mo_sep');
if max([nt_sel_dB0(:); nt_sel_mo(:)]) > numel(nt_sel) || min([nt_sel_dB0(:); nt_sel_mo(:)]) < 1
    error('motion_newton_round:ntSelSepRange', ...
        'nt_sel_dB0/nt_sel_mo (%s / %s) fall outside the %d exported echoes %s', ...
        mat2str(nt_sel_dB0), mat2str(nt_sel_mo), numel(nt_sel), mat2str(nt_sel));
end
fprintf(['[sepmo] echoes exported nt_sel = %s | B0 sub-fit uses echo(s) %s ' ...
         '(position %s) | motion sub-fit uses echo(s) %s (position %s)\n'], ...
        mat2str(nt_sel), mat2str(nt_sel(nt_sel_dB0)), mat2str(nt_sel_dB0), ...
        mat2str(nt_sel(nt_sel_mo)), mat2str(nt_sel_mo));

% --- echo selection actually used, recorded with the results -------------
% nt_sel       : absolute echo numbers exported to the solver
% nt_mo/nt_dB0 : absolute echo numbers each sub-fit used
% nt_sel_mo/nt_sel_dB0 : the same, as positions within nt_sel
nt_mo  = nt_sel(nt_sel_mo);
nt_dB0 = nt_sel(nt_sel_dB0);

if has_mo_variant
    coilmo_suffix = ['_coilmo',num2str(size(SEs_tmp_mo,1))];
else
    coilmo_suffix = '';
end
if st_flag
    sep_name_tail = ['_st1',sep_block,joint_block,run_tag,suffix];
    st1_vars = {'smooth_flag','rotcoil_flag','lam_joint','basis_sel','nt_dB0_joint'};
else
    sep_name_tail = ['_st0',sep_block,run_tag,suffix];
    st1_vars = {};
end
save_sep = [file_series,data_label_sep,'_',save_base,'_coil',num2str(Ncoils),coilmo_suffix,sep_name_tail]
% a cache under the previous naming is reused if it provably matches; that
% name did not record nt_dB0 or the joint echoes, so check what it stored
save_sep_legacy = [file_series,data_label_sep,'_',save_base,'_coil',num2str(Ncoils),coilmo_suffix,'_nt',num2str(nt_sel(1)),'_',num2str(nt_sel(end)),'_lam',num2str(lam_joint),'_basis',num2str(numel(basis_sel)),'_sB',num2str(smooth_flag),rotcoil_flag,'_ntmo',num2str(nt_sel_mo(1)),num2str(nt_sel_mo(end)),'_st',num2str(st_flag),st_joint_tag,sepmo_tag,run_tag,suffix];
if ~exist([dataset_path,save_sep,'.mat'],'file') && ...
        legacy_ok([dataset_path,save_sep_legacy,'.mat'], struct('nt_mo',nt_mo_abs,'nt_dB0',nt_dB0_abs), ...
                  sep_is_default || ~isempty(run_tag), ...
                  ~st_flag || joint_is_default || ~isempty(run_tag))
    fprintf('[sepmo] reusing cache under its old name %s\n', save_sep_legacy);
    save_sep = save_sep_legacy;
end

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
if has_mo_variant
    Ncoils_mo = size(SEs_tmp_mo,1);  % SEs_tmp_mo already permuted [4 1 2 3] like SEs_tmp
    kspace_tmp_mo = zeros(nbins_sep,Ncoils_mo,Nt_sel,fid_sel,Nx_nav);
end

% kx_tmp = [ceil(Nx_nav/5+1):ceil(Nx_nav/5*2),ceil(Nx_nav/5*3+1):ceil(Nx_nav/5*4)];
kx_tmp = [ceil(Nx_nav/5+1):ceil(Nx_nav/5*4)];

ratios_sep = zeros(1,nbins_sep);
for ii_bin = 1:nbins_sep
    % fid_int marks the first fid_sel shots of every fid_sel+fid_dis group,
    % so the discarded ones are left out here. Without it a bin holds
    % fid_sel+fid_dis shots while kspace_tmp/ky_tmp/kz_tmp are only
    % fid_sel wide, and MATLAB would silently grow them -- exporting the
    % very shots fid_dis asked to drop. With fid_dis = 0 fid_int is all
    % true, so this changes nothing for the default.
    ii_shot = find(resp_bin_sep == ii_bin & fid_int);
    shot_per_bin = numel(ii_shot);
    for ii = 1:shot_per_bin
        ii_all = ii_shot(ii);
        ii_out = floor((ii_all-1)/shot_fid)+1;
        ii_in  = mod(ii_all-1,shot_fid)+1;

        ratios_sep(ii_bin) = ratios_all(ii_all);
        kspace_tmp(ii_bin,:,:,ii,:) = permute(fftc(kspace_cart_orig_fid{ii_out}{ii_in}(:,nt_sel,:),1),[3 2 1]);
        if has_mo_variant
            kspace_tmp_mo(ii_bin,:,:,ii,:) = permute(fftc(kspace_cart_orig_fid_mo{ii_out}{ii_in}(:,nt_sel,:),1),[3 2 1]);
        end

        ky_tmp(ii_bin,ii,:) = ky_sel(fid_idx(ii_all),:);
        kz_tmp(ii_bin,ii,:) = kz_sel(fid_idx(ii_all),:);
    end
end

writecfl([dataset_sep_path,'ky_tmp'],ky_tmp);
writecfl([dataset_sep_path,'kz_tmp'],kz_tmp);
writecfl([dataset_sep_path,'kx_tmp'],kx_tmp);

writecfl([dataset_sep_path,'kspace_all_new_tmp'],kspace_tmp);
if has_mo_variant
    writecfl([dataset_sep_path,'kspace_all_new_tmp_mo'],kspace_tmp_mo);
    writecfl([dataset_sep_path,'SEs_tmp_mo'],SEs_tmp_mo);
end

writecfl([dataset_sep_path,'ratios_new_tmp'],ratios_sep)
a0_tmp = permute(a_nav_refs(:,:,:,1,nt_sel),[5 1 2 3 4]);
writecfl([dataset_sep_path,'a0_tmp'],a0_tmp);

TEs_tmp = TEs_fid(nt_sel);
writecfl([dataset_sep_path,'TEs_tmp'],TEs_tmp);
writecfl([dataset_sep_path,'nt_sel_mo'],nt_sel_mo);
writecfl([dataset_sep_path,'nt_sel_dB0'],nt_sel_dB0);

writecfl([dataset_sep_path,'traj_bin'],fid_idx_in);
% write B0 baseline from different motionstate, optional

% this part is same for all different methods
% writecfl([dataset_sep_path,'coil_sel_tmp'],coil_sel);
writecfl([dataset_sep_path,'Bas_tmp'],Bas_tmp);

writecfl([dataset_sep_path,'SEs_tmp'],SEs_tmp);

writecfl([dataset_sep_path,'Basn_tmp'],Basn_tmp);

clear motion_init
for ii_bin = 1:nbins_sep
    %     motion_init(ii_bin,:) = mean(motion_est_all((fid_sel+fid_dis)*(ii_bin-1)+(1:fid_sel+fid_dis),:),1);
    motion_init(ii_bin,:) = mean(motion_est_all(resp_bin_sep == ii_bin & fid_int,:),1);
end
writecfl([dataset_sep_path,'motion_init'],motion_init); % -1 is prepared for python

% motion initialization

motion_coef_init = motion_init(:,basis_sel);
motion_coef_init = cat(2,ones(nbins_sep,1),motion_coef_init);

% B0 initialization
% st_flag alone is enough to gate this now: any st=1 failure (too few
% motion-state bins, the joint B0 script failing, or its result containing
% NaN -- see st1_failed above) already returns early, before this point,
% with the entire st=1 round skipped. So reaching here with st_flag==1
% means the joint B0 estimation succeeded and B0_coef_test_est/nbasis/
% motion_weight_test/B0_coef_smooth are valid.
if st_flag
    disp('B0 init')
    if ~smooth_flag
        B0_proj_init = reshape(reshape(B0_coef_test_est,[],nbasis)*(motion_coef_init/sqrtm(motion_weight_test))',[size_nav nbins_sep]);
    else
        B0_proj_init = reshape(reshape(B0_coef_smooth,[],nbasis)*(motion_coef_init/sqrtm(motion_weight_test))',[size_nav nbins_sep]);
    end
    writecfl([dataset_sep_path,'B0_base_tmp'],permute(B0_proj_init,[4 1 2 3])); % -1 is prepared for python
else
    disp('no B0 bg')
    B0_proj_init = 0
end
%%

if ~exist([dataset_path,save_sep,'.mat'],'file')

    display('sep new batch')
    whichdev = choose_GPU();
    % 1. Define the path to your Miniforge environment's libstdc++
    pyenv_ = motion_python_env(); miniforge_lib = pyenv_.libstdcxx;

    % 2. Construct the command, prepending the LD_PRELOAD env variable
    % NW/edit 20260910: also force LD_LIBRARY_PATH -- see the identical
    % note earlier in this file (motion_est_b_coef_dyn_app.py call).
    nvidia_lib = pyenv_.nvidia;
    python_path = sepmo_python
    % only the fast solver understands --n_alt; 1 is its default anyway
    if sepmo_n_alt ~= 1
        alt_arg = [' --n_alt ', num2str(sepmo_n_alt)];
    else
        alt_arg = '';
    end

    cmd = [ ...
        'LD_PRELOAD=', miniforge_lib, ' ', ...
        'LD_LIBRARY_PATH=', nvidia_lib, ' ', ...
        pyenv_.bin, ' ', python_path, ...
        ' --path ', dataset_sep_path, ...
        ' --gpu ', num2str(whichdev-1) ...
        alt_arg ...
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

    %% diverged per-shot fit: the python solver zeroes bins that went NaN, and
    %% a diverging bin explodes (e.g. 1e31). If most bins are like that the
    %% estimate is useless -- for st=1 return [] so motion_newton_round falls
    %% back to the st=0 result (same as the joint-B0 failure path above).
    out_row = sum(abs(motion_est),2)>60 | sum(abs(cr_est),2)>1000;
    bad_row = out_row | ~any(motion_est,2);
    if isequal(p.st_flag,1) && mean(bad_row) > 0.5
        warning('motion_newton_round:sepDiverged', ...
            'per-shot motion fit diverged (%d of %d bins invalid) -- treating st=1 as failed for this round.', ...
            nnz(bad_row), numel(bad_row));
        r = [];
        return
    end

    %% first line removing: replace outlier rows by the nearest valid row
    %% (previous one, or the next one for leading outliers) -- the old
    %% tmpl-1 indexing failed when row 1 was an outlier and copied one
    %% outlier into the next when two were adjacent
    good = find(~out_row);
    if ~isempty(good)
        for t = find(out_row).'
            src = good(find(good < t, 1, 'last'));
            if isempty(src)
                src = good(find(good > t, 1));
            end
            motion_est(t,:) = motion_est(src,:);
            cr_est(t,:)     = cr_est(src,:);
        end
    end
    %% balance
    motion_est_new = reshape(motion_est,lcm(shot_fid,fid_sel+fid_dis)/(fid_sel+fid_dis),[],6);
    cr_est_new = reshape(cr_est,lcm(shot_fid,fid_sel+fid_dis)/(fid_sel+fid_dis),[],9);
    for i = 1:lcm(shot_fid,fid_sel+fid_dis)/(fid_sel+fid_dis)
        correction = mean(motion_est_new(i,:,:)-mean(motion_est_new,1),2);
        motion_est_new(i,:,:) = motion_est_new(i,:,:)-correction;

        correction = mean(cr_est_new(i,:,:)-mean(cr_est_new,1),2);
        cr_est_new(i,:,:) = cr_est_new(i,:,:)-correction;
    end
    motion_est_new = reshape(motion_est_new,[],6);
    cr_est_new = reshape(cr_est_new,[],9);

    %%
    motion_est_wfilt = medfilt1(motion_est_new,lcm(shot_fid,(fid_sel+fid_dis))/(fid_sel+fid_dis));
    cr_est_wfilt = medfilt1(cr_est_new,lcm(shot_fid,(fid_sel+fid_dis))/(fid_sel+fid_dis));

    %% save
    save([dataset_path,save_sep],'motion_est','motion_est_wfilt','cr_est','cr_est_wfilt','Bas','motion_params_base*','B0_nav_antiregi','B0_scout_antiregi','voxel_size','mask_B0_nav','ratios','ratios_sep','a_nav_refs','TEs_fid','fid_idx','fid_sel','fid_dis','ky_tmp','kz_tmp','resp_bin_sep','B0_proj_init','nt_sel','nt_mo','nt_dB0','nt_sel_mo','nt_sel_dB0','st_flag',st1_vars{:})
else
    load([dataset_path,save_sep])
end
%% save for correction
save_data = ['motion_est_',save_base,'_coil',num2str(Ncoils),coilmo_suffix,sep_name_tail,'.mat'];
motion_est1 = motion_est_wfilt .*[1 1 1 voxel_size];
B0_est_2nd = reshape(Bas_norm*(cr_est_wfilt.'),[size_nav size(cr_est_wfilt,1)]);
B0_est_2nd = single(B0_est_2nd);
save([file_path,save_data],'motion_est1','B0_est_2nd','B0_proj_init','resp_bin_sep', ...
     'nt_sel','nt_mo','nt_dB0','nt_sel_mo','nt_sel_dB0','st_flag',st1_vars{:})

%% joint quasi newton
% motion_est_qnj = readcfl([dataset_sep_path,'motion_est_reg3']);
% cr_est_qnj     = readcfl([dataset_sep_path,'cr_est_reg3']);
% save([dataset_path,'estimation_',file_name,'_',scout_flag,'_coil',num2str(Ncoils),'_sB',num2str(smooth_flag),'_nt',num2str(nt_sel(1)),num2str(nt_sel(end)),'_ntmo',num2str(nt_sel_mo(1)),num2str(nt_sel_mo(end)),'_sel',num2str(fid_sel),'_basis',num2str(numel(basis_sel)),'_st',num2str(st_flag),'_qnj',rotcoil_flag,'.mat'],'motion_est_qnj','cr_est_qnj','nbins_sep','resp_bin_sep')
% motion_est1 = motion_est_qnj .*[1 1 1 voxel_size];
% B0_est_2nd = reshape(Bas_norm*(cr_est_qnj.'),[size_nav size(cr_est_qnj,1)]);
% B0_est_2nd = single(B0_est_2nd);
% save([file_path,'motion_params_',scout_flag,'_coil',num2str(Ncoils),'_sB',num2str(smooth_flag),'_nt',num2str(nt_sel(1)),num2str(nt_sel(end)),'_ntmo',num2str(nt_sel_mo(1)),num2str(nt_sel_mo(end)),'_sel',num2str(fid_sel),'_basis',num2str(numel(basis_sel)),'_st',num2str(st_flag),'_qnj',rotcoil_flag,'.mat'],'motion_est1','B0_est_2nd','B0_proj_init','resp_bin_sep')

%% delete
rmdir(dataset_sep_path,'s')
if st_flag && ~keep_joint_dir
    rmdir(dataset_joint_path,'s')
end

r = struct();
r.basis_sel        = basis_sel;
r.motion_est_wfilt = motion_est_wfilt;
r.cr_est_wfilt     = cr_est_wfilt;

end

function v = p_get(p, name, default)
% Field of p if it is present and non-empty, otherwise default.
if isfield(p, name) && ~isempty(p.(name))
    v = p.(name);
else
    v = default;
end
end

function pos = nt_abs2pos(want_abs, nt_sel, name)
% Map absolute echo numbers onto their positions inside nt_sel, which is
% the convention motion_est_lbfgs*.py expects for nt_sel_mo / nt_sel_dB0.
want_abs = want_abs(:).';
pos = zeros(1, numel(want_abs));
for k = 1:numel(want_abs)
    hit = find(nt_sel == want_abs(k), 1);
    if isempty(hit)
        error('motion_newton_round:ntNotExported', ...
            ['%s asks for echo %d, which is not in nt_sel = %s. Add it to ' ...
             'nt_sel_sep (the echoes handed to the per-shot solver).'], ...
            name, want_abs(k), mat2str(nt_sel));
    end
    pos(k) = hit;
end
end

function [mo_abs, dB0_abs] = sep_echoes(ScanType, pos_path, mo_in, dB0_in)
% Absolute echoes of the per-shot motion and B0 sub-fits: the
% sequence-dependent defaults unless overridden.
if strcmp(ScanType, 'fs') && (contains(pos_path,'fsm17') || contains(pos_path,'fsm16'))
    nt_sel0 = [1:4 7:8];
else
    nt_sel0 = 1:6;
end
mo_abs  = nt_sel0(3);
dB0_abs = nt_sel0([1 5]);
if ~isempty(mo_in),  mo_abs  = mo_in(:).';  end
if ~isempty(dB0_in), dB0_abs = dB0_in(:).'; end
end

function s = echo_label(e)
% [1 4 5 6] -> '1456'
s = sprintf('%d', e);
end

function ok = legacy_ok(f, want, ok_if_missing, ok_unrecorded)
% True if the old-format cache f exists and matches: every field of want
% that f stored must equal it, a field it did not store is accepted only if
% ok_if_missing, and ok_unrecorded covers what the file cannot show at all.
ok = ok_unrecorded && isfile(f);
if ~ok, return, end
vars = who('-file', f);
fn = fieldnames(want);
for k = 1:numel(fn)
    if ismember(fn{k}, vars)
        v = load(f, fn{k});
        ok = ok && isequal(v.(fn{k})(:).', want.(fn{k})(:).');
    else
        ok = ok && ok_if_missing;
    end
end
end
