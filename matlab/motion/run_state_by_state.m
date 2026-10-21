% Public entry point: state-by-state motion + dB0 estimation.
%
% This release covers ONE path through the pipeline -- the per-shot
% ("state-by-state") joint motion/B0 fit solved by
%   python/motion_est_lbfgs_fast.py   batched Levenberg-Marquardt solver
%
% The spatiotemporal-B0 (st = 1) path is NOT part of this release; st_flag is
% fixed to 0 below and should stay that way.
%
%   dataset : folder holding the series and the scout
%   series  : the motion series folder
%   scout   : full path to the scout recon .mat
addpath(genpath(fileparts(fileparts(mfilename('fullpath')))));  % matlab/

dataset = '/path/to/dataset/';
series  = [dataset 'XcXX_bkm15_20_54'];
scout   = [dataset 'XcXX_bkfs_6_9_match/recon/recon_XcXX_Ax_None_ave1_recon.mat'];

opts = struct();
opts.st_flag        = 0;          % state-by-state only -- do not change
opts.solver         = 'fast';     % motion_est_lbfgs_fast.py
opts.grad_solver    = 'fast';
opts.n_alt          = 2;          % B0 <-> motion alternations (needs 'fast')
opts.motion_init    = 'grad';     % or 'zero' to skip the gradient stage
opts.fid_sel        = 4;          % shots per motion state

% echo selection, absolute echo numbers; the exported set is their union
opts.nt_sel_mo_sep  = [1 3];
opts.nt_sel_dB0_sep = [1 4 5 6];

motion_test_newton_bkm(dataset, series, scout, opts);
