function r = motion_newton_round_slurm(varargin)
% MOTION_NEWTON_ROUND_SLURM  run motion_newton_round as a slurm job.
%
% motion_newton_round.m calls out to motion_est_lbfgs.py / choose_GPU() for
% every value of p.st_flag (0, 1, or 'both' -- which runs both st=0 and st=1
% internally in a single call), so it always needs an actual CUDA device. On
% a SLURM-managed machine (dhall), GPUs are typically only visible inside a
% real SLURM GPU allocation (srun/sbatch --gres=gpu:1) -- being on a machine
% that has GPUs isn't enough if the calling MATLAB session itself (e.g. a
% plain interactive session) wasn't launched inside one. This wraps the
% whole motion_newton_round call in one such allocation, the same way
% recon_calib_low_slurm.m does for recon_calib_low.
%
% Outer call (5 inputs, made from the interactive MATLAB session):
%   r = motion_newton_round_slurm(p, motion_est_intp2, basis_sel, recompute_basis_sel, suffix)
%   same signature as motion_newton_round itself. Packages the inputs to
%   disk, submits a slurm job that opens a new MATLAB to actually run
%   motion_newton_round on a GPU node, blocks until that job finishes, then
%   loads and returns the r it saved. Uses p.dataset_path to place the
%   slurm_jobs/ folder.
%
% Inner call (1 input, made by the job's own MATLAB instance):
%   motion_newton_round_slurm(args_file)
%   loads (p, motion_est_intp2, basis_sel, recompute_basis_sel, suffix) from
%   args_file, runs motion_newton_round, and saves r to the result_file
%   recorded in args_file. The caller appends "; exit" so MATLAB closes
%   itself when done, which in turn lets the wrapping srun command -- and so
%   the slurm job -- finish.

slurm_partition = 'small';
slurm_gres = 'gpu:1';
slurm_time = '11:59:00';

if nargin == 5
    %% ---- outer call: package inputs and submit the slurm job ----
    [p, motion_est_intp2, basis_sel, recompute_basis_sel, suffix] = deal(varargin{:});
    dataset_path = p.dataset_path;

    job_dir = fullfile(dataset_path, 'slurm_jobs');
    if ~exist(job_dir, 'dir')
        mkdir(job_dir);
    end
    tag = datestr(now, 'yyyymmdd_HHMMSSFFF');
    args_file = fullfile(job_dir, ['motion_newton_round_args_', tag, '.mat']);
    result_file = fullfile(job_dir, ['motion_newton_round_result_', tag, '.mat']);
    save(args_file, 'p', 'motion_est_intp2', 'basis_sel', 'recompute_basis_sel', 'suffix', 'result_file', '-v7.3');

    % a plain shell launcher avoids quoting headaches inside srun
    job_script = fullfile(job_dir, ['motion_newton_round_job_', tag, '.sh']);
    job_log = fullfile(job_dir, ['motion_newton_round_job_', tag, '.log']);
    main_path = fileparts(fileparts(mfilename('fullpath')));  % .../code

    % the matlab install differs per compute node, and the node srun lands
    % on is only known once the job starts -- so pick the binary by
    % hostname inside the script itself rather than on the submitting host
    fid = fopen(job_script, 'w');
    fprintf(fid, '#!/bin/bash\n');
    fprintf(fid, 'host=$(hostname)\n');
    fprintf(fid, 'case "$host" in\n');
    fprintf(fid, '  *kikyo*) matlab_bin=/usr/local/MATLAB/R2022b/bin/matlab ;;\n');
    fprintf(fid, '  *dhall*) matlab_bin=/usr/local/MATLAB/R2023a/bin/matlab ;;\n');
    fprintf(fid, '  *) matlab_bin=matlab ;;\n');
    fprintf(fid, 'esac\n');
    % -batch, NOT -r. With -r an uncaught error aborts the statement list
    % BEFORE the trailing exit runs, and MATLAB drops to its interactive
    % prompt; under srun there is no tty, so it waits on stdin forever and the
    % job sits there until --time expires -- the error is in the log but
    % nothing ever stops. Reproduced: `-r "error('boom'); exit"` with stdin
    % open printed the error, then '>>' , and had to be killed at 90s.
    % -batch never goes interactive and exits 1 on an uncaught error (0 on
    % success, both verified on R2023a), so srun propagates that and the
    % `status ~= 0` check below actually fires.
    fprintf(fid, '"$matlab_bin" -batch "addpath(genpath(''%s/Funcs'')); addpath(genpath(''%s/motion'')); if exist(''setup_cirEPTI_path'',''file''), setup_cirEPTI_path(); end; motion_newton_round_slurm(''%s'')"\n', main_path, main_path, args_file);
    fclose(fid);
    fileattrib(job_script, '+x');

    slurm_cmd = sprintf('srun -p %s --gres=%s --time=%s --output=%s bash %s', ...
        slurm_partition, slurm_gres, slurm_time, job_log, job_script);

    disp(['[motion_newton_round_slurm] submitting: ', slurm_cmd])

    % system() forks the shell with MATLAB's OWN current directory as its
    % starting cwd; if that's gone stale (e.g. an NFS automount that timed
    % out), the spawned shell fails at its own getcwd() before srun even runs.
    % cd into job_dir (just freshly created above, so guaranteed to exist
    % right now) to give MATLAB -- and therefore the forked shell -- a
    % known-good cwd. Same fix as recon_calib_low_slurm.m's system() call.
    cd(job_dir);
    status = system(slurm_cmd);
    cd(dataset_path);

    slurm_job_check(status, job_log, result_file, 'motion_newton_round_slurm job');

    out = load(result_file, 'r');
    r = out.r;

    delete(args_file);
    delete(result_file);

elseif nargin == 1
    %% ---- inner call: this runs inside the slurm allocation ----
    args_file = varargin{1};
    in = load(args_file);
    r = motion_newton_round(in.p, in.motion_est_intp2, in.basis_sel, in.recompute_basis_sel, in.suffix);
    save(in.result_file, 'r', '-v7.3');

else
    error('motion_newton_round_slurm:badNargin', 'expected 1 or 5 input arguments, got %d', nargin);
end
end
