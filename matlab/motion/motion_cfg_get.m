function v = motion_cfg_get(name, fallback)
%MOTION_CFG_GET  One machine-specific path, from (in order):
%   1. the environment  MOTION_DATA_ROOT / BART_TOOLBOX_PATH
%   2. motion_local_config.m, if present (gitignored; see the .example)
%   3. the fallback given by the caller
envname = struct('data_root','MOTION_DATA_ROOT','bart_path','BART_TOOLBOX_PATH', ...
                 'python_bin','MOTION_PYTHON');
v = '';
if isfield(envname, name), v = getenv(envname.(name)); end
if isempty(v) && exist('motion_local_config','file') == 2
    try
        cfg = motion_local_config();
        if isfield(cfg, name), v = cfg.(name); end
    catch
    end
end
if isempty(v), v = fallback; end
end
