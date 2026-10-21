function motion_test_newton_bkm(dataset_path, dataseries_path, scout_path_arg, opts)
% MOTION_TEST_NEWTON_BKM  motion/B0 estimation for one EPTI data series.
%
% Standalone (all args omitted): behaves exactly as before -- uigetdir for
% the dataset and data series, and (see the scout-selection block below)
% offers a dialog to pick an external/bkfs scout even when an internal one
% was found, falling back to the internal one on Cancel.
%
% Called from cirEPTI_recon.m (all 3 args given): no prompts at all --
% dataset_path/dataseries_path/scout_path_arg are used directly, exactly as
% resolved by the caller (bkfs scout preferred over internal scout there).
%
% opts (4th argument, optional) overrides anything in the USER OPTIONS
% block below; leaving it out reproduces the previous behaviour exactly.
% Examples:
%
%   % default: reference solver, st = 'both'
%   motion_test_newton_bkm(ds, series, scout)
%
%   % fast per-shot solver (~40x), results tagged '_fast' so they sit
%   % alongside the reference ones instead of overwriting them
%   motion_test_newton_bkm(ds, series, scout, struct('solver','fast'))
%
%   % motion only, no spatiotemporal B0, single round
%   motion_test_newton_bkm(ds, series, scout, struct('st_flag',0))
%
%   % no gradient step at all -- per-shot fit starts from zero motion
%   motion_test_newton_bkm(ds, series, scout, ...
%       struct('motion_init','zero','st_flag',0,'solver','fast'))
%
%   % two B0<->motion cycles instead of the single pass
%   motion_test_newton_bkm(ds, series, scout, ...
%       struct('solver','fast','n_alt',2))
%
%   % both stages on the fast path
%   motion_test_newton_bkm(ds, series, scout, ...
%       struct('solver','fast','grad_solver','fast'))
%
%   % change the shot binning (fewer shots per motion state = finer states)
%   motion_test_newton_bkm(ds, series, scout, struct('fid_sel',6,'fid_dis',0))
%
%   % pick the echoes used by the per-shot fit
%   motion_test_newton_bkm(ds, series, scout, ...
%       struct('nt_sel_sep',1:6, 'nt_sel_dB0_sep',[1 5], 'nt_sel_mo_sep',[2 3]))
%
%   % fully interactive (all three paths from the usual dialogs) but with
%   % options -- pass the struct on its own:
%   motion_test_newton_bkm(struct('solver','fast','st_flag',0))
if nargin < 4
    opts = struct();
end
if nargin < 3
    scout_path_arg = [];
end
if nargin < 2
    dataseries_path = [];
end
if nargin < 1
    dataset_path = [];
end

% Options-only form: motion_test_newton_bkm(opts) -- pick the dataset, the
% data series and the scout from the dialogs as usual, but still set the
% options.
if isstruct(dataset_path)
    if ~isempty(dataseries_path) || ~isempty(scout_path_arg)
        error('motion_test_newton_bkm:badArgs', ...
            'the options-only form takes the struct as its only argument');
    end
    opts = dataset_path;
    dataset_path = [];
end
if isempty(opts)
    opts = struct();
end

% Whether to prompt is decided by whether we were actually told where the
% data is, NOT by nargin -- otherwise passing opts would silently suppress
% the dialogs. So motion_test_newton_bkm([],[],[],opts) is still fully
% interactive, while cirEPTI_recon.m's 3-argument call keeps its existing
% behaviour: no prompts, and an error if its scout_path_arg is empty.
called_programmatically = ~isempty(dataset_path) && ~isempty(dataseries_path);

%% ======================= USER OPTIONS ===================================
% Each default below is used only when the corresponding field of `opts` is
% absent or empty, so everything here can equally be set from the caller.

% --- which per-shot motion/B0 solver to run ------------------------------
%   'original' - motion_est_lbfgs.py        (reference, unchanged)
%   'fast'     - motion_est_lbfgs_fast.py   (~40x; same inputs and outputs)
% The two write to different result files ('_fast' suffix), so switching
% never overwrites or silently reloads the other one's cache.
opts = opt_default(opts, 'solver', 'fast');

% --- which gradient-stage (step 1) implementation to run -----------------
%   'original' - motion_grad_est_spatial.m        (reference, unchanged)
%   'fast'     - motion_grad_est_spatial_fast.m   (~10x; same outputs)
% As with solver, the two write to different result files ('_gfast'), so
% switching never overwrites or silently reloads the other one's cache.
opts = opt_default(opts, 'grad_solver', 'fast');

% --- which joint-B0 (st=1 spatiotemporal B0) implementation to run --------
%   'original' - motion_est_b_coef_dyn_app.py        (reference, unchanged)
%   'fast'     - motion_est_b_coef_dyn_app_fast.py   (same inputs and outputs)
% The two write to different jointB0 cache files ('_jfast' suffix).
opts = opt_default(opts, 'joint_solver', 'fast');
% true keeps the jointB0 export directory (the python inputs) after the run
opts = opt_default(opts, 'keep_joint_dir', false);
% appended to the temporary sepmo/ and jointB0/ export folder names only, so
% several runs on the same series (e.g. a sweep) do not share them
opts = opt_default(opts, 'export_tag', '');
opts = opt_default(opts, 'second_round', false);

% --- coil selection ------------------------------------------------------
%   'split'   - SVD-compress coil_sel_n_svd channels for joint-B0 and the
%               lbfgs motion sub-fit, and hard-select coil_sel_n_b0hard
%               (or all, if fewer) for the B0 sub-fit and gradient stage.
%               'combined' is accepted as a synonym.
%   'uniform' - one hard-threshold reduction shared by every step
%   'none'    - no coil reduction
% opts.coil_selection_type is accepted as an alias for opts.coil_sel_mode.
if isfield(opts,'coil_selection_type') && ~isempty(opts.coil_selection_type) ...
        && (~isfield(opts,'coil_sel_mode') || isempty(opts.coil_sel_mode))
    opts.coil_sel_mode = opts.coil_selection_type;
end
opts = opt_default(opts, 'coil_sel_mode', 'split');

% --- where the per-shot motion fit starts from --------------------------
%   'grad' - warm start from the gradient stage's m_est (step 1), as the
%            pipeline has always done
%   'zero' - start every motion state at zero. The gradient stage's only
%            downstream product IS this warm start (its cr_est_poly and
%            c_est are saved but never read), so 'zero' skips step 1
%            entirely and goes straight to the per-shot fit.
% Note: the st=1 spatiotemporal B0 decomposition locates motion-stable
% segments by looking for *changes* in this initial estimate, so with
% 'zero' it has nothing to work with and will report too few bins and fall
% back to st=0. Use 'zero' with st_flag = 0 unless you know otherwise.
opts = opt_default(opts, 'motion_init', 'grad');

% Basis for the gradient stage's polynomial B0 fit:
%   'norm' - Bas_norm, the [1 10 10 10 100 ...] scaled basis the rest of
%            the pipeline uses (cond 7.6 vs 687 -- see
%            motion_grad_est_spatial_fast.m). cr_est_poly then shares units
%            with the per-shot stage. Fitted field is unchanged.
%   'raw'  - Bas, as motion_grad_est_spatial.m has always used.
% Only the 'fast' grad implementation honours this.
opts = opt_default(opts, 'grad_basis', 'norm');

% --- B0 <-> motion alternation inside the per-shot solver ----------------
% The solver fits B0 with the motion frozen, then motion with that B0
% frozen. n_alt = 1 (the original) does that once, so the B0 never sees the
% motion correction the second half found -- measured on this data, the
% motion moves ~3.8 deg rms after B0 is fixed. n_alt > 1 repeats the pair,
% each sub-fit warm starting from its own previous answer. 'fast' solver
% only.
opts = opt_default(opts, 'n_alt', 2);

% Free-form string appended to the per-shot result file names. The built-in
% name records only nt_sel(1), nt_sel(end) and nt_sel_mo(1), nt_sel_mo(end),
% and nothing of nt_sel_dB0 -- so e.g. nt_mo [1 3] and [1 2 3] collide, and
% every nt_dB0 choice collides. Set a tag when sweeping those.
opts = opt_default(opts, 'tag', '');

% [] = decide from the hostname (nested srun on dhall). Set false when you
% are already inside a GPU allocation and want the stages called directly
% -- a sweep of many runs otherwise pays a MATLAB respawn and a full
% serialisation of p for every single one.
opts = opt_default(opts, 'use_slurm', []);

% --- spatiotemporal mode, handed to motion_newton_round ------------------
%   0      - motion only, via the per-shot solver, no B0 init
%   1      - spatiotemporal B0 decomposition first, then the per-shot fit
%   'both' - st = 0, then st = 1 warm-started from it
opts = opt_default(opts, 'st_flag', 'both');

% --- navigator -----------------------------------------------------------
% nav_type is read from the fid recon file unless given here. The old
% navigators (nav_type < 17) do not support the spatiotemporal B0
% decomposition, so for those st_flag is forced to 0 and only a single
% round is run. Set legacy_nav_force_st0 = false to run them as asked
% anyway.
opts = opt_default(opts, 'nav_type', []);
opts = opt_default(opts, 'legacy_nav_force_st0', true);
% Which acquisitions carry the fid shots (see the fid_idx block below).
% [] = decide from nav_type / ScanType; true = take the nav_int == 1 shots;
% false = index the shots directly.
opts = opt_default(opts, 'use_nav_int', []);

% --- shot binning --------------------------------------------------------
% fid_sel shots are kept and the next fid_dis discarded, repeating, so each
% motion state is built from fid_sel shots out of every fid_sel+fid_dis.
% fid_sel is best an integer multiple of shot_fid_in. Raising
% fid_sel+fid_dis past shot_fid makes the trajectory wrap, so a bin then
% contains repeated (ky,kz) lines -- of which the solver keeps only the
% LAST and discards the rest (it does not average them); you get a warning
% if that is about to happen.
% [] = pick from nav_type once it is known (see below): 4 for the old
% navigators (nav_type < 17), 12 for nav_type >= 17.
opts = opt_default(opts, 'fid_sel', []);
if ~isfield(opts,'fid_dis') || isempty(opts.fid_dis)
    opts.fid_dis = 0;   % not opt_default: 0 is a meaningful value here
end

% --- echo (nt) selection -------------------------------------------------
% All of these are ABSOLUTE 1-based echo numbers in the fid echo train
% (this dataset has 6). [] keeps the sequence-dependent default the code
% picks on its own.
%   *_grad  : step 1, the linear gradient update (motion_grad_est_spatial)
%   *_sep   : the per-shot fit (motion_est_lbfgs*.py)
%   *_joint : the spatiotemporal B0 decomposition (motion_est_b_coef_dyn_app.py)
%
% Within each stage the three are coupled, so change them together:
%   nt_sel_*     which echoes that stage works with
%   nt_sel_dB0_* which of them the B0 sub-fit uses
%   nt_sel_mo_*  which of them the motion sub-fit uses
% For *_sep, nt_sel_dB0_sep / nt_sel_mo_sep must be a subset of nt_sel_sep
% (they are converted to positions inside it for the python side, which is
% the convention it expects -- see motion_newton_round.m); you get a clear
% error if they are not. For *_grad, nt_sel_dB0_grad and nt_sel_mo_grad are
% what drive the maths and need not be a subset of anything, while
% nt_sel_grad only labels the cached result file.
opts = opt_default(opts, 'nt_sel_grad',     []);
opts = opt_default(opts, 'nt_sel_dB0_grad', []);
opts = opt_default(opts, 'nt_sel_mo_grad',  []);
opts = opt_default(opts, 'nt_sel_sep',      []);
opts = opt_default(opts, 'nt_sel_dB0_sep',  [1 4 5 6]);
opts = opt_default(opts, 'nt_sel_mo_sep',   [1 3]);
opts = opt_default(opts, 'nt_sel_joint',    []);
opts = opt_default(opts, 'nt_sel_dB0_joint',[]);
%% ========================================================================

%% setup path
if isunix
    sep = '/';
else
    sep = '\';
end
% all code paths are relative to this file, so the tree can be moved
here      = fileparts(mfilename('fullpath'));   % .../code/motion
main_path = fileparts(here);                    % .../code
% BART is not needed on the state-by-state path (bart() is only called by
% the st=1 joint-B0 stage, which this release does not include).

% motion operator and registration/recon helpers (local copy, was an
% external user's tree; code only, no data -- see motion_package/README.md)
addpath(genpath(fullfile(here, 'motion_package')))

addpath(genpath(fullfile(main_path, 'Funcs')))
% addpath(genpath(fullfile(main_path, 'sLORAKS')))
addpath(genpath(here))
% addpath(fullfile(bart_path, 'matlab'));
% setenv('TOOLBOX_PATH',bart_path);

addpath(genpath(strcat(main_path, sep, 'cirEPTI')))
% addpath(fullfile(bart_path, 'matlab'));

%% check whether this system can submit slurm jobs
% GPUs on dhall are typically only visible inside an actual SLURM GPU
% allocation (srun/sbatch --gres=gpu:1) -- being on a machine that has GPUs
% isn't enough if this interactive session itself wasn't launched inside
% one. motion_grad_est_spatial (gpuArray) and motion_newton_round
% (motion_est_lbfgs.py / motion_est_b_coef_dyn_app.py via choose_GPU) both
% always use the GPU, so every call to either is routed through their
% *_slurm wrappers when on dhall.
[~, hostname] = system('hostname');
hostname = strtrim(hostname);
use_slurm = strcmp(hostname, 'dhall');   % may be overridden by opts.use_slurm below

%%
% in this code, try to register everything to scout

%% set up function

vec = @(x)x(:);


%% data loading
% dataset_path = [data_path, '<study>/<exam>/'];
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
if isempty(dataset_path)
    data_path = motion_cfg_get('data_root', pwd)
    dataset_path = uigetdir(data_path, 'which dataset');
    dataset_path = [dataset_path,filesep];
end
if ~endsWith(dataset_path, filesep)
    dataset_path = [dataset_path, filesep];
end

if isempty(dataseries_path)
    dataseries_path = uigetdir(dataset_path, 'select the data series path to work on');
end

% no need of full path anymore, 260814
% full_path = uigetdir(dataset_path, 'select the calibration path');
% tmp_path = [full_path,filesep,'recon',filesep,'fits_60*'];
% tmp_path = dir(tmp_path);
% full_path = [tmp_path.folder, filesep, tmp_path.name];
% clear tmp_path

pos_path = [dataseries_path,filesep,'recon',filesep,'recon_*data_sigpy_recon.mat'];
tmp_path = dir(pos_path)

if isempty(tmp_path) % old way of saving data
    pos_path = [dataseries_path,filesep,'recon',filesep,'recon_*_recon.mat'];
    tmp_path = dir(pos_path);
    pos_path = [tmp_path.folder, filesep, tmp_path.name];

    pos_nav_path = [];
    no_nav = false; % [] here means the old "Standard" (single pos_data file) mode, not the new no-nav path
    scout_path_internal = '';

    pos_fid_path = [dataseries_path,filesep,'recon',filesep,'recon_*_sigpy.mat'];
    tmp_path = dir(pos_fid_path);
    pos_fid_path = [tmp_path.folder, filesep, tmp_path.name];

    % in old way, scout alwasy separate. Honour scout_path_arg when the
    % caller supplied one (it may be the recon .mat itself or the series
    % folder), so these datasets can be run without the dialog too.
    if ~isempty(scout_path_arg)
        if endsWith(scout_path_arg, '.mat')
            scout_path = scout_path_arg;
        else
            tmp_path = dir([scout_path_arg,filesep,'recon',filesep,'recon_*_recon.mat']);
            if isempty(tmp_path)
                error('motion_test_newton_bkm:noScoutMat', ...
                    'no recon_*_recon.mat under %s', scout_path_arg);
            end
            scout_path = [tmp_path(1).folder,filesep,tmp_path(1).name];
        end
    else
        scout_path = uigetdir(dataset_path, 'select scout path');
        tmp_path = dir([scout_path,filesep,'recon',filesep,'recon_*_recon.mat'])
        scout_path = [tmp_path.folder,filesep,tmp_path.name];
    end

else  % new way of saving data

    pos_path = [tmp_path.folder, filesep, tmp_path.name];

    % Nav is not critical to motion estimation -- it may genuinely be
    % absent (e.g. switched off via the Siemens WIP alFree(13) toggle in
    % partition_kspace.m, so no recon_*nav_sigpy_recon.mat ever gets
    % produced). Detect that explicitly rather than blindly indexing into
    % a possibly-empty dir() result, which would silently build a garbage
    % path (just filesep) instead of erroring or working correctly.
    nav_path = [dataseries_path,filesep,'recon',filesep,'recon_*nav_sigpy_recon.mat'];
    tmp_path = dir(nav_path);
    no_nav = isempty(tmp_path);
    if no_nav
        disp('no nav recon found -- running motion estimation without nav data')
        pos_nav_path = [];
    else
        pos_nav_path = [tmp_path.folder, filesep, tmp_path.name];
    end

    fid_path = [dataseries_path,filesep,'recon',filesep,'recon_*fid_sigpy.mat'];
    tmp_path = dir(fid_path);
    pos_fid_path = [tmp_path.folder, filesep, tmp_path.name];


    scout_path_glob = [dataseries_path,filesep,'recon',filesep,'recon_*scout_sigpy_recon.mat'];
    tmp_path = dir(scout_path_glob);
    if isempty(tmp_path)
        scout_path_internal = '';
    else
        scout_path_internal = [tmp_path.folder, filesep, tmp_path.name];
    end

    if called_programmatically
        % already decided by the caller (cirEPTI_recon.m already preferred
        % bkfs over internal scout, or passed the internal one, or -- if
        % scout_path_arg is empty -- there was nothing to use at all) --
        % no prompting here.
        scout_path = scout_path_arg;
    else
        % standalone: always offer the external/bkfs choice, even when an
        % internal scout was found -- Cancel falls back to the internal
        % scout if there is one.
        if isempty(scout_path_internal)
            scout_prompt = 'select external scout/bkfs path (no internal scout found)';
        else
            scout_prompt = 'select external scout/bkfs path (Cancel to use internal scout)';
        end
        scout_dir = uigetdir(dataset_path, scout_prompt);
        if ~isequal(scout_dir, 0)
            tmp_path = dir([scout_dir,filesep,'recon',filesep,'recon_*_recon.mat']);
            scout_path = [tmp_path(1).folder, filesep, tmp_path(1).name];
        elseif ~isempty(scout_path_internal)
            scout_path = scout_path_internal;
        else
            error('motion_test_newton_bkm:noScout', 'no scout available (no internal scout, none selected)');
        end
    end
end

scout_path = erase(scout_path, dataset_path)
pos_path   = erase(pos_path, dataset_path)
pos_fid_path   = erase(pos_fid_path, dataset_path)
% full_path  = erase(full_path, dataset_path)
try
    pos_nav_path   = erase(pos_nav_path, dataset_path)
end
if ~isempty(scout_path_internal)
    scout_path_internal = erase(scout_path_internal, dataset_path);
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
cfg.no_nav             = no_nav;
cfg.scout_path_internal = scout_path_internal;
% cfg.full_path    = full_path;
cfg.fminunc_path    = [main_path,filesep,'motion'];


if contains(pos_path,'fsm17') || contains(pos_path,'fsm16')
    cfg.Nt_ahead    = 0;
    cfg.Nt_scout_in = [1:6 27:32 55:60];
else
    cfg.Nt_ahead    = 7;
    cfg.Nt_scout_in = 6;
end
cfg.bin_center      = 1;
cfg.nt_pick         = 6;
cfg.scout_rep_sel   = 1;  % for crm18,19,20, select 1
cfg.shot_fid_gap    = 1;

file_name   = regexp(cfg.pos_path, filesep, 'split');
file_name = file_name{1};
file_series = regexp(file_name, '^\d+', 'match', 'once');

if ~exist([dataset_path,file_series,'data_',scout_flag,'.mat'],'file')
    out = motion_newton_core(cfg);

    save([dataset_path,file_series,'data_',scout_flag,'.mat'],'out','cfg','*_path','*_flag','-v7.3')
else
    load([[dataset_path,file_series,'data_',scout_flag,'.mat']])
end
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
Ncoils_full = Ncoils;  % captured before any coil_sel_flag reduction, for the
                        % 'split' mode's "keep all if original < N" clamp

Bas_norm = Bas.*[1 10 10 10 100 100 100 100 100];

%% estimation parameters flag
st_flag = opts.st_flag;   % spatialtemporal flag, passed to motion_newton_round:
               %   0      - motion-only est via the per-shot solver, no B0 init
               %   1      - est B0_init, then motion est with B0_init
               %   'both' - run st = 0 first, then st = 1 warm-started from the st = 0 result

%% navigator type -> whether the spatiotemporal B0 decomposition is usable
% nav_type is stored in the fid recon file. Reading it from there (rather
% than re-deriving it from the sequence name) keeps this in step with
% config_navtype.m, which is what wrote it.
if isempty(opts.nav_type)
    nav_type = [];
    try
        tmp_nav = load([dataset_path, pos_fid_path], 'nav_type');
        if isfield(tmp_nav,'nav_type') && ~isempty(tmp_nav.nav_type)
            nav_type = double(tmp_nav.nav_type(1));
        end
        clear tmp_nav
    catch ME
        warning('motion_test_newton_bkm:navTypeRead', ...
            'could not read nav_type from %s (%s)', pos_fid_path, ME.message);
    end
else
    nav_type = double(opts.nav_type);
end

single_round = false;
if isempty(nav_type)
    % Unknown: do NOT silently change what the caller asked for.
    warning('motion_test_newton_bkm:navTypeUnknown', ...
        ['nav_type not found -- running with st_flag as requested. Pass ' ...
         'opts.nav_type to be explicit.']);
    nav_type = NaN;
elseif nav_type < 17 && opts.legacy_nav_force_st0
    % old navigator: no spatiotemporal B0 decomposition, one round only
    if ~isequal(st_flag, 0)
        fprintf(['[motion_test_newton_bkm] nav_type = %g (< 17, old navigator): ' ...
                 'forcing st_flag = 0 and a single round (was %s)\n'], ...
                nav_type, mat2str(st_flag));
    end
    st_flag = 0;
    single_round = true;
end
fprintf('[motion_test_newton_bkm] nav_type = %g, st_flag = %s, solver = %s\n', ...
        nav_type, mat2str(st_flag), opts.solver);

%% per-shot solver selection
switch lower(opts.solver)
    case 'original'
        sepmo_python = [main_path, filesep, 'epti_python', filesep, ...
                        'mr_recon-main', filesep, 'motion_est_lbfgs.py'];
        sepmo_tag    = '';
    case 'fast'
        sepmo_python = [main_path, filesep, 'epti_python', filesep, ...
                        'mr_recon-main', filesep, 'motion_est_lbfgs_fast.py'];
        sepmo_tag    = '_fast';   % keeps both solvers' results side by side
        if opts.n_alt ~= 1
            sepmo_tag = [sepmo_tag,'_alt',num2str(opts.n_alt)];
        end
    otherwise
        error('motion_test_newton_bkm:badSolver', ...
              'opts.solver must be ''original'' or ''fast'', got ''%s''', opts.solver);
end
if ~isscalar(opts.n_alt) || opts.n_alt < 1 || mod(opts.n_alt,1) ~= 0
    error('motion_test_newton_bkm:badNAlt', 'opts.n_alt must be a positive integer');
end
if opts.n_alt ~= 1 && strcmpi(opts.solver,'original')
    error('motion_test_newton_bkm:altNeedsFast', ...
        ['opts.n_alt = %d needs the fast solver -- motion_est_lbfgs.py has no ' ...
         'alternation. Set opts.solver = ''fast''.'], opts.n_alt);
end
if ~exist(sepmo_python, 'file')
    error('motion_test_newton_bkm:noSolver', 'solver script not found: %s', sepmo_python);
end

%% joint-B0 implementation
switch lower(opts.joint_solver)
    case 'original'
        joint_python = [main_path, filesep, 'epti_python', filesep, ...
                        'mr_recon-main', filesep, 'motion_est_b_coef_dyn_app.py'];
        joint_tag    = '';
    case 'fast'
        joint_python = [main_path, filesep, 'epti_python', filesep, ...
                        'mr_recon-main', filesep, 'motion_est_b_coef_dyn_app_fast.py'];
        joint_tag    = '_jfast';
    otherwise
        error('motion_test_newton_bkm:badJointSolver', ...
              'opts.joint_solver must be ''original'' or ''fast'', got ''%s''', opts.joint_solver);
end
if ~exist(joint_python, 'file')
    error('motion_test_newton_bkm:noJointSolver', 'joint-B0 script not found: %s', joint_python);
end

%% gradient-stage (step 1) implementation
switch lower(opts.grad_solver)
    case 'original'
        grad_fast = false;  grad_tag = '';
    case 'fast'
        grad_fast = true;   grad_tag = '_gfast';
        if strcmpi(opts.grad_basis,'raw')
            grad_tag = [grad_tag,'_braw'];  % keep the two bases' caches apart
        end
    otherwise
        error('motion_test_newton_bkm:badGradSolver', ...
              'opts.grad_solver must be ''original'' or ''fast'', got ''%s''', opts.grad_solver);
end
switch lower(opts.motion_init)
    case 'grad', skip_grad = false; init_tag = '';
    case 'zero', skip_grad = true;  init_tag = '_m0';
    otherwise
        error('motion_test_newton_bkm:badMotionInit', ...
              'opts.motion_init must be ''grad'' or ''zero'', got ''%s''', opts.motion_init);
end
if skip_grad
    grad_tag = init_tag;    % no gradient stage ran, so its tag is meaningless
    if ~isequal(st_flag, 0)
        warning('motion_test_newton_bkm:zeroInitWithSt', ...
            ['motion_init = ''zero'' with st_flag = %s: the st=1 B0 ' ...
             'decomposition needs motion variation in the initial estimate ' ...
             'and will most likely fall back to st=0.'], mat2str(st_flag));
    end
else
    grad_tag = [grad_tag, init_tag];
end
fprintf('[motion_test_newton_bkm] grad_solver = %s, motion_init = %s\n', ...
        opts.grad_solver, opts.motion_init);
if ~isempty(opts.use_slurm)
    use_slurm = logical(opts.use_slurm);
    fprintf('[motion_test_newton_bkm] use_slurm forced to %d\n', use_slurm);
end
if isempty(opts.fid_sel)        % nav_type-dependent default
    if ~isnan(nav_type) && nav_type < 17
        opts.fid_sel = 4;
    else
        opts.fid_sel = 12;
    end
    fprintf('[motion_test_newton_bkm] fid_sel defaulted to %d for nav_type = %g\n', ...
            opts.fid_sel, nav_type);
end
second_round_flag = opts.second_round;
fid_sel = opts.fid_sel; % shots kept per motion state (see USER OPTIONS)
smooth_flag = 1;  % mainly for some cases
fid_dis = opts.fid_dis; % shots discarded after each kept run
if ~isscalar(fid_sel) || fid_sel < 1 || mod(fid_sel,1) ~= 0 || ...
   ~isscalar(fid_dis) || fid_dis < 0 || mod(fid_dis,1) ~= 0
    error('motion_test_newton_bkm:badFidSel', ...
        'fid_sel must be a positive integer and fid_dis a non-negative integer (got %s / %s)', ...
        mat2str(fid_sel), mat2str(fid_dis));
end
if mod(fid_sel, shot_fid_in) ~= 0
    warning('motion_test_newton_bkm:fidSelNotMultiple', ...
        'fid_sel = %d is not an integer multiple of shot_fid_in = %d', fid_sel, shot_fid_in);
end
if fid_sel + fid_dis > shot_fid
    warning('motion_test_newton_bkm:fidBinWraps', ...
        ['fid_sel + fid_dis = %d exceeds shot_fid = %d, so the trajectory wraps and ' ...
         'each bin will contain repeated (ky,kz) lines. The solver keeps only the LAST ' ...
         'of each repeat and discards the others -- it does not average them.'], ...
        fid_sel + fid_dis, shot_fid);
end
% The balancing/median-filter steps later on reshape the per-bin estimates
% into lcm(shot_fid, fid_sel+fid_dis)/(fid_sel+fid_dis) rows, which only
% works if the bin count divides by that. Catch it here with a usable
% message instead of a bare reshape error 30 minutes into the run.
fid_grp   = fid_sel + fid_dis;
fid_nbins = floor((nshot_total-1)/fid_grp) + 1;
fid_rows  = lcm(shot_fid, fid_grp) / fid_grp;
if mod(fid_nbins, fid_rows) ~= 0
    error('motion_test_newton_bkm:fidBinReshape', ...
        ['fid_sel+fid_dis = %d gives %d motion states, which is not divisible by ' ...
         'lcm(shot_fid=%d, %d)/%d = %d -- the later balancing reshape would fail. ' ...
         'Values that always work are those dividing shot_fid: %s (other ' ...
         'sums can work too, e.g. 16 for shot_fid 36 -- the test above is exact).'], ...
        fid_grp, fid_nbins, shot_fid, fid_grp, fid_grp, fid_rows, ...
        mat2str(divisors_of(shot_fid)));
end
fprintf('[binning] fid_sel = %d kept, fid_dis = %d discarded, of shot_fid = %d -> %d motion states\n', ...
        fid_sel, fid_dis, shot_fid, fid_nbins);
switch lower(opts.coil_sel_mode)
    case 'none',                 coil_sel_flag = 0; coil_sel_mode = 'uniform';
    case 'uniform',              coil_sel_flag = 1; coil_sel_mode = 'uniform';
    case {'split','combined'},   coil_sel_flag = 1; coil_sel_mode = 'split';
    otherwise
        error('motion_test_newton_bkm:badCoilSel', ...
            'opts.coil_sel_mode must be ''split'', ''uniform'' or ''none'', got ''%s''', ...
            opts.coil_sel_mode);
end
% modes: 'uniform' (one hard-threshold reduction
                            % shared by every downstream step) or 'split'
                            % (SVD-compress coil_sel_n_svd channels for
                            % joint-B0 + the lbfgs motion sub-fit; hard-select
                            % coil_sel_n_b0hard channels, or all if the
                            % dataset has fewer, for the lbfgs B0 sub-fit and
                            % gradient estimation)
coil_sel_method = 'head';  % 'whole' (legacy: energy over the whole FOV) or
                            % 'head' (energy over a shoulder-excluded mask --
                            % see coil_sel below) -- 'uniform' mode only
coil_keep_frac = [];       % fraction of coils to keep, by weight rank
                            % (e.g. 0.5 keeps the top half). [] falls back to
                            % the legacy threshold coil_weight>1/sqrt(Ncoils).
                            % -- 'uniform' mode only
coil_sel_n_svd = 16;        % 'split' mode only
coil_sel_n_b0hard = 32;     % 'split' mode only
rotcoil_flag = '_rotcoil';
lam_joint = 1e-9;
basis_sel = [1:6];  % select which motion params as basis, by default is [2:3]

%% coil selection
% Ranks coils by how much sensitivity-map energy they contribute, then
% either keeps every coil above the legacy equal-share threshold or the
% top coil_keep_frac fraction by rank -- either way over one of two
% energy definitions:
%   'whole' - the original behaviour: energy over the entire FOV. A coil
%             whose main lobe sits over the shoulder/neck (present in the
%             FOV for scans that don't crop it) can pass this even though
%             it contributes little to the head signal motion estimation
%             actually needs.
%   'head'  - energy restricted to a soft head-only mask along the SI axis
%             (dim 3 of SEs_perm, matching a_nav_refs' [Ny,Nz,Nx,...]
%             convention), from the same detect_and_mask_shoulders
%             heuristic motion_newton_core.m already applies to the scout
%             image itself -- so a shoulder-heavy coil now ranks by how
%             much it actually sees of the head, not the shoulder.
if coil_sel_flag && strcmpi(coil_sel_mode, 'split')
    disp('coil_sel (split: SVD for joint-B0/motion, hard for B0/grad-est)')
    [SEs_svd, ksp_svd] = coil_compress(SEs_perm, kspace_cart_orig_fid, coil_sel_n_svd, 'svd');
    n_b0 = min(coil_sel_n_b0hard, Ncoils_full);
    [SEs_hard_b0, ksp_hard_b0] = coil_compress(SEs_perm, kspace_cart_orig_fid, n_b0, 'hard');

    coil_sel_split_active = true;
    SEs_perm_svd = SEs_svd; kspace_svd = ksp_svd; Ncoils_svd = size(SEs_svd,4);
    clear SEs_svd ksp_svd

    % becomes the primary set for everything downstream that isn't
    % joint-B0/lbfgs-motion: gradient estimation and the lbfgs B0 sub-fit
    SEs_perm = SEs_hard_b0;
    kspace_cart_orig_fid = ksp_hard_b0;
    Ncoils = size(SEs_hard_b0,4);
    clear SEs_hard_b0 ksp_hard_b0
elseif coil_sel_flag
    disp('coil_sel')
    if strcmpi(coil_sel_method, 'head')
        mask_si_coilsel = detect_and_mask_shoulders(abs(a_nav_refs(:,:,:,1,1)), 3);
        SEs_for_weight = SEs_perm .* reshape(mask_si_coilsel, 1, 1, []);
    else
        SEs_for_weight = SEs_perm;
    end
    coil_weight = sqrt(sum(abs(reshape(SEs_for_weight,[],Ncoils)).^2,1)) / norm(SEs_for_weight(:));

    if isempty(coil_keep_frac)
        coil_sel = find(coil_weight > 1/sqrt(Ncoils));
    else
        n_keep = max(1, round(coil_keep_frac * Ncoils));
        [~, coil_order] = sort(coil_weight, 'descend');
        coil_sel = sort(coil_order(1:n_keep));
    end

    SEs_perm = SEs_perm(:,:,:,coil_sel);

    for ii = 1:size(kspace_cart_orig_fid,2)
        for jj = 1:size(kspace_cart_orig_fid{1},2)
            kspace_cart_orig_fid{ii}{jj} = kspace_cart_orig_fid{ii}{jj}(:,:,coil_sel);
        end
    end
    Ncoils = numel(coil_sel);
end

save_base = [scout_flag,'_sel',num2str(fid_sel),grad_tag];
if fid_dis > 0
    % only appended when non-zero, so every existing fid_dis = 0 result
    % file keeps exactly the name it has today
    save_base = [save_base,'_dis',num2str(fid_dis)];
end

%% prepare B0 basis and SEs for python
Bas_tmp = reshape(Bas,Ny_nav,Nz_nav,Nx_nav,9);
Bas_tmp = permute(Bas_tmp,[4 1 2 3]);
Basn_tmp = Bas_tmp.*([1 10 10 10 100 100 100 100 100].') ;

SEs_tmp = permute(SEs_perm,[4 1 2 3]);
%%
% Which acquisitions are the fid shots, and what trajectory index each one
% carries. The trajectory cycles with period shot_fid over the FULL
% acquisition list, so if anything other than fid shots is interleaved, the
% kept shots' indices are not simply 1,2,3,...
%   nav_type 17  - kspace_cart_orig_fid already holds only the data's fid
%                  shots, so the running shot number is the index.
%   nav_type <17 - other acquisitions are interleaved; the fid shots are the
%                  nav_int == 1 ones and their index comes from their
%                  position in the full list. Same as the GE 'fs' case that
%                  was already handled here.
% opts.use_nav_int forces either branch.
if isempty(opts.use_nav_int)
    use_nav_int = (strcmp(out.ScanType,'fs') && ~contains(dataset_path,'Siemens')) ... % GE fs, as before
               || (~isnan(nav_type) && nav_type < 17);                                 % old navigators
else
    use_nav_int = logical(opts.use_nav_int);
end

if use_nav_int
    if isempty(out.nav_int)
        error('motion_test_newton_bkm:noNavInt', ...
            ['nav_int is needed to pick the fid shots (nav_type = %g) but is ' ...
             'empty in the saved data. Set opts.use_nav_int = false to index ' ...
             'the shots directly instead.'], nav_type);
    end
    kept = find(out.nav_int(:).' == 1);        % absolute acquisition positions
    fprintf(['[fid_idx] nav_int == 1 selects %d fid shots out of %d ' ...
             'acquisitions (nshot_total = %d)\n'], ...
            numel(kept), numel(out.nav_int), nshot_total);
    if numel(kept) < nshot_total
        error('motion_test_newton_bkm:tooFewNavInt', ...
            ['nav_int == 1 gives only %d fid shots but nshot_total is %d.'], ...
            numel(kept), nshot_total);
    end
    kept = kept(1:nshot_total);
    fid_idx = mod(kept-1, shot_fid) + 1;       % true trajectory index of each

    % kspace_cart_orig_fid may be stored over EVERY acquisition, while every
    % downstream consumer indexes it by the running fid-shot counter:
    %     ii_out = floor((ii_all-1)/shot_fid)+1;  ii_in = mod(ii_all-1,shot_fid)+1
    % Left as-is that reads the navigator acquisitions as if they were fid
    % shots and never reaches the last ones. Pull out the nav_int == 1 shots
    % and regroup them so the counter indexes them directly.
    n_held = numel(kspace_cart_orig_fid) * numel(kspace_cart_orig_fid{1});
    if n_held >= max(kept) && n_held > nshot_total
        knew = cell(1, nshot_total/shot_fid);
        for ii_a = 1:numel(knew)
            inner = cell(1, shot_fid);
            for ii_b = 1:shot_fid
                pp = kept((ii_a-1)*shot_fid + ii_b);
                inner{ii_b} = kspace_cart_orig_fid{floor((pp-1)/shot_fid)+1}{mod(pp-1,shot_fid)+1};
            end
            knew{ii_a} = inner;
        end
        kspace_cart_orig_fid = knew; clear knew inner
        fprintf(['[fid_idx] kspace_cart_orig_fid held %d shots (all acquisitions); ' ...
                 'subset to the %d fid shots\n'], n_held, nshot_total);
    elseif n_held == nshot_total
        fprintf('[fid_idx] kspace_cart_orig_fid already holds only the %d fid shots\n', n_held);
    else
        error('motion_test_newton_bkm:kspaceShotMismatch', ...
            ['kspace_cart_orig_fid holds %d shots, which matches neither the %d ' ...
             'fid shots nor the %d acquisitions nav_int describes.'], ...
            n_held, nshot_total, numel(out.nav_int));
    end
else
    fid_idx = 1:nshot_total;
    fid_idx = mod(fid_idx-1, shot_fid) + 1;
end

% (fid_idx is already the per-shot trajectory index, set above)

fid_idx_in = floor((fid_idx-1)/shot_fid_in)+1;



fid_int = repmat([ones(1,fid_sel),zeros(1,fid_dis)],[1 ceil(nshot_total/(fid_sel+fid_dis))]);
fid_int = logical(fid_int(1:nshot_total));

if strcmp(out.ScanType, 'fs') && (contains(pos_path,'fsm17') || contains(pos_path,'fsm16'))
    nt_sel = [1 3 7];
    nt_sel_dB0 = [1 7]
    nt_sel_mo  = [1 3]
else
    nt_sel = [1 3 5];
    nt_sel_dB0 = [1 5]
    nt_sel_mo  = [1 3]
end
% step-1 (linear gradient update) overrides -- see the USER OPTIONS block.
% Here nt_sel_dB0/nt_sel_mo index the echo train directly (unlike the
% per-shot stage, where they index into nt_sel -- motion_newton_round.m
% converts for you).
if ~isempty(opts.nt_sel_dB0_grad); nt_sel_dB0 = opts.nt_sel_dB0_grad; end
if ~isempty(opts.nt_sel_mo_grad);  nt_sel_mo  = opts.nt_sel_mo_grad;  end
% nt_sel is DERIVED: the union of what the two sub-fits need. Supply
% nt_sel_dB0_grad / nt_sel_mo_grad only; nt_sel_grad is still honoured.
if ~isempty(opts.nt_sel_grad)
    nt_sel = opts.nt_sel_grad;
else
    nt_sel = union(nt_sel_dB0, nt_sel_mo);
end
nt_total = numel(TEs_fid);
bad = [nt_sel(:); nt_sel_dB0(:); nt_sel_mo(:)];
bad = bad(bad < 1 | bad > nt_total);
if ~isempty(bad)
    error('motion_test_newton_bkm:ntGradRange', ...
        ['gradient-step echo index %d is outside the echo train (1..%d). ' ...
         'nt_sel=%s nt_sel_dB0=%s nt_sel_mo=%s'], bad(1), nt_total, ...
        mat2str(nt_sel), mat2str(nt_sel_dB0), mat2str(nt_sel_mo));
end
fprintf(['[grad step] echoes nt_sel = %s | B0 est uses %s | ' ...
         'motion est uses %s  (of %d echoes)\n'], ...
        mat2str(nt_sel), mat2str(nt_sel_dB0), mat2str(nt_sel_mo), nt_total);

resp_bin = floor(((1:nshot_total)-1)/ (fid_sel + fid_dis) )+1; % this helps map from fid group to all fid trajectory


%% use GPU
% useGPU=(exist('gpuNUFFT','file')>1); %true if using gpuNUFFT. false if using irt.
useGPU=true;
% if useGPU
%     try
%         whichdev = choose_GPU(); %Give MATLAB main software access to the GPU before the gpuNUFFT software locks it up
%     catch
%         gpudev = gpuDevice(3);
%     end
% end
%% prepare for estimation with gradient, thisestimation gives an answer every shot_fid_in * shot_fid_gap
% absolute echoes run together, e.g. [1 3] -> _mo13; same labels as sepmo
save_grad = [file_series, 'grad_',save_base,'_coil',num2str(Ncoils),'_mo',sprintf('%d',nt_sel_mo),'_s',sprintf('%d',nt_sel_dB0)];
% a cache under the previous naming (_nt<a><b>_ntmo<a><b>, which did not
% record the B0 echoes) is reused if what it stored matches
save_grad_legacy = [file_series, 'grad_',save_base,'_coil',num2str(Ncoils),'_nt',num2str(nt_sel(1)),num2str(nt_sel(end)),'_ntmo',num2str(nt_sel_mo(1)),num2str(nt_sel_mo(end))];
if ~exist([dataset_path,save_grad,'.mat'],'file') && ...
        grad_legacy_ok([dataset_path,save_grad_legacy,'.mat'], nt_sel_mo, nt_sel_dB0, isempty(opts.nt_sel_dB0_grad))
    fprintf('[grad step] reusing cache under its old name %s\n', save_grad_legacy);
    save_grad = save_grad_legacy;
end
motion_str = out;
motion_str.kspace_cart_orig_fid = kspace_cart_orig_fid;
motion_str.SEs_perm = SEs_perm;
motion_str.Ncoils = Ncoils;
motion_str.Bas_norm = Bas_norm;      % conditioned basis for the poly fit
motion_str.grad_basis = opts.grad_basis;


timing_str.fid_idx    = fid_idx;
timing_str.fid_idx_in = fid_idx_in;
timing_str.resp_bin   = resp_bin;
timing_str.fid_int    = fid_int;  % which shot to keep or to discard
timing_str.nt_sel_dB0 = nt_sel_dB0;  % dB0 est
timing_str.nt_sel_mo  = nt_sel_mo;  % motion est

% init_str.B0_est_spatial = B0_est_poly;
init_str = struct();

if skip_grad
%% start every motion state at zero -- step 1 is skipped (see USER OPTIONS)
fprintf('[motion_test_newton_bkm] skipping the gradient stage, zero motion init\n');
motion_est_intp2 = zeros(fid_nbins,6);
else
if ~exist([dataset_path,save_grad,'.mat'],'file')
    motion_str.grad_fast = grad_fast;   % read by the slurm wrapper too
    if use_slurm
        est_out = motion_grad_est_spatial_slurm(motion_str, timing_str, init_str, dataset_path);
    else
        if useGPU
        whichdev = choose_GPU()
        end
        if grad_fast
            est_out = motion_grad_est_spatial_fast(motion_str, timing_str,init_str);
        else
            est_out = motion_grad_est_spatial(motion_str, timing_str,init_str);
        end
    end

    % nt_sel_dB0 = est_out.nt_sel_dB0;  % dB0 est
    % nt_sel     = est_out.nt_sel;  % motion est

    % save
    m_est = est_out.m_est;
    cr_est_poly = est_out.cr_est_poly;
    c_est = est_out.c_est;
    save([dataset_path,save_grad],'m_est','c_est','cr_est_poly','Bas','motion_params_base','B0_nav_antiregi','B0_scout_antiregi','voxel_size','mask_B0_nav','ratios','nt_sel*')
else
    load([dataset_path,save_grad])
end
%%
m_est_new = reshape(m_est,lcm(shot_fid,fid_sel+fid_dis)/(fid_sel+fid_dis),[],6);
% cr_est_poly_new = reshape(cr_est_poly,lcm(shot_fid,fid_sel+fid_dis)/(fid_sel+fid_dis),[],9);
for i = 1:lcm(shot_fid,fid_sel+fid_dis)/(fid_sel+fid_dis)
    correction = mean(m_est_new(i,:,:)-mean(m_est_new,1),2);
    m_est_new(i,:,:) = m_est_new(i,:,:)-correction;

    % correction = mean(cr_est_poly_new(i,:,:)-mean(cr_est_poly_new,1),2);
    % cr_est_poly_new(i,:,:) = cr_est_poly_new(i,:,:)-correction;

end
m_est_new = reshape(m_est_new,[],6);
% cr_est_poly_new = reshape(cr_est_poly_new,[],9);
%% interp for better smoothes motion
motion_est_intp2 = medfilt1(m_est_new,lcm(shot_fid,fid_sel+fid_dis)/(fid_sel+fid_dis));
end
% motion_est_all = reshape(repmat(reshape(motion_est_intp2,1,numel(fid_idx_in),6),[shot_fid_in 1 1]),[],6);

%% bundle the context shared by every estimation round (see motion_newton_round.m)
p = struct();
p.dataset_path         = dataset_path;
p.file_name             = file_name;
p.ScanType              = out.ScanType;
p.pos_path              = pos_path;
p.file_series           = file_series;
p.file_path             = file_path;
p.save_base             = save_base;
p.Ncoils                = Ncoils;
p.lam_joint             = lam_joint;
p.smooth_flag           = smooth_flag;
p.rotcoil_flag          = rotcoil_flag;
p.ky_fid                = ky_fid;
p.kz_fid                = kz_fid;
p.Nx_nav                = Nx_nav;
p.Ny_nav                = Ny_nav;
p.Nz_nav                = Nz_nav;
p.flip_flag             = flip_flag;
p.ratios_all            = ratios_all;
p.ratios                = ratios;
p.nshot_total           = nshot_total;
p.shot_fid              = shot_fid;
p.kspace_cart_orig_fid  = kspace_cart_orig_fid;
p.fid_idx               = fid_idx;
p.fid_idx_in            = fid_idx_in;
p.a_nav_refs            = a_nav_refs;
p.TEs_fid               = TEs_fid;
p.Bas_tmp               = Bas_tmp;
p.SEs_tmp               = SEs_tmp;
p.Basn_tmp              = Basn_tmp;
p.B0_nav_antiregi       = B0_nav_antiregi;
p.B0_scout_antiregi     = B0_scout_antiregi;
p.nav_rep               = nav_rep;
p.motion_params_base    = motion_params_base;
p.size_nav              = size_nav;
p.st_flag               = st_flag;
p.fid_sel               = fid_sel;
p.fid_dis               = fid_dis;
p.Bas                   = Bas;
p.Bas_norm              = Bas_norm;
p.voxel_size            = voxel_size;
p.mask_B0_nav           = mask_B0_nav;
p.sepmo_python          = sepmo_python;
p.sepmo_tag             = sepmo_tag;   % solver + n_alt only
p.run_tag               = opts.tag;    % goes at the end of the sepmo/jointB0/motion_est names; the gradient cache stays shared
p.sepmo_n_alt           = opts.n_alt;
p.joint_python          = joint_python;
p.joint_tag             = joint_tag;
p.keep_joint_dir        = opts.keep_joint_dir;
p.export_tag            = opts.export_tag;
p.nt_sel_joint          = opts.nt_sel_joint;
p.nt_sel_dB0_joint      = opts.nt_sel_dB0_joint;
p.nt_sel_sep            = opts.nt_sel_sep;
p.nt_sel_dB0_sep        = opts.nt_sel_dB0_sep;
p.nt_sel_mo_sep         = opts.nt_sel_mo_sep;

if exist('coil_sel_split_active','var') && coil_sel_split_active
    p.SEs_tmp_jointb0              = permute(SEs_perm_svd,[4 1 2 3]);
    p.kspace_cart_orig_fid_jointb0 = kspace_svd;
    p.Ncoils_jointb0               = Ncoils_svd;
    p.SEs_tmp_mo                   = p.SEs_tmp_jointb0;  % same SVD variant, reused for lbfgs motion
    p.kspace_cart_orig_fid_mo      = kspace_svd;
end

%% round 1: initial motion/B0 estimation
if use_slurm
    round1 = motion_newton_round_slurm(p, motion_est_intp2, basis_sel, true, '');
else
    round1 = motion_newton_round(p, motion_est_intp2, basis_sel, true, '');
end

%% optional second round: repeat with a different warm start
% (skipped for the old navigator, which is single-round by definition)
if second_round_flag && ~single_round
    p.st_flag = 1;  % for 2nd round, no need to do st0
    if use_slurm
        round2 = motion_newton_round_slurm(p, round1.motion_est_wfilt, round1.basis_sel, false, '_2nd');
    else
        round2 = motion_newton_round(p, round1.motion_est_wfilt, round1.basis_sel, false, '_2nd');
    end
end
end

function ok = grad_legacy_ok(f, nt_mo, nt_dB0, dB0_is_default)
% True if the old-format gradient cache f exists and used these echoes. A
% file that did not store them is accepted only for the default B0 echoes,
% the one thing its name could not show.
ok = isfile(f);
if ~ok, return, end
vars = who('-file', f);
if all(ismember({'nt_sel_mo','nt_sel_dB0'}, vars))
    v = load(f, 'nt_sel_mo', 'nt_sel_dB0');
    ok = isequal(v.nt_sel_mo(:).', nt_mo(:).') && isequal(v.nt_sel_dB0(:).', nt_dB0(:).');
else
    ok = dB0_is_default;
end
end

function opts = opt_default(opts, name, default)
% Fill in a USER OPTIONS field only if the caller did not set it.
if ~isfield(opts, name) || isempty(opts.(name))
    opts.(name) = default;
end
end

function d = divisors_of(n)
% The fid_sel+fid_dis values that always keep the balancing reshape valid.
d = find(mod(n, 1:n) == 0);
end
