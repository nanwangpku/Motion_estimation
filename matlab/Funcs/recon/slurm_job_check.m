function slurm_job_check(status, job_log, result_file, ctx)
%SLURM_JOB_CHECK  Turn a finished srun call into a MATLAB error, log included.
%
%   slurm_job_check(status, job_log)
%   slurm_job_check(status, job_log, result_file)
%   slurm_job_check(status, job_log, result_file, ctx)
%
% status      exit code from system(srun ...)
% job_log     the job's --output file
% result_file (optional) file the job was supposed to write; '' to skip
% ctx         (optional) short string naming the job, for the message
%
% Two failure modes, both of which used to surface badly:
%
%   1. The job errored. The srun exit code is non-zero, but the caller only
%      reported "see <log>", so the actual MATLAB error stayed in a file
%      nobody had open. The log tail is now inlined into the error.
%
%   2. The job exited 0 but wrote no result. load() then failed with an
%      opaque "Unable to read file" pointing at a temp .mat, rather than
%      saying the job had not produced anything.
%
% NOTE this only reports a failure that has already happened. A job that
% HANGS is a separate problem, fixed at the source: the job scripts launch
% MATLAB with -batch, not -r, so an uncaught error exits 1 instead of
% dropping to an interactive prompt that waits on stdin until --time.

if nargin < 3, result_file = ''; end
if nargin < 4 || isempty(ctx), ctx = 'slurm job'; end

tail_txt = local_tail(job_log, 40);

if status ~= 0
    error('slurm_job_check:jobFailed', ...
        ['%s failed (srun exit code %d).\n' ...
         '  log: %s\n' ...
         '  --- last lines of the log ---\n%s'], ...
        ctx, status, job_log, tail_txt);
end

if ~isempty(result_file) && ~exist(result_file, 'file')
    error('slurm_job_check:noResult', ...
        ['%s reported success (exit 0) but did not write its result file.\n' ...
         '  expected: %s\n' ...
         '  log: %s\n' ...
         '  --- last lines of the log ---\n%s'], ...
        ctx, result_file, job_log, tail_txt);
end
end


function txt = local_tail(f, n)
% last n lines of f, as one string; never throws -- this runs on the error
% path, where a second failure would hide the first
txt = '(log unavailable)';
try
    if ~exist(f, 'file'), txt = sprintf('(no log at %s)', f); return, end
    fid = fopen(f, 'r');
    if fid < 0, return, end
    c = onCleanup(@() fclose(fid));
    lines = {};
    while true
        l = fgetl(fid);
        if ~ischar(l), break, end
        lines{end+1} = l; %#ok<AGROW>
    end
    if isempty(lines), txt = '(log is empty)'; return, end
    lines = lines(max(1, numel(lines)-n+1):end);
    txt = sprintf('  %s\n', lines{:});
catch
    % keep the default
end
end
