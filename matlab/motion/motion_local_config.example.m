function cfg = motion_local_config()
%MOTION_LOCAL_CONFIG  Machine-specific paths. Copy to motion_local_config.m
%                     and edit; that copy is gitignored.
%
% Everything else in this repo resolves relative to its own location, so this
% is the only file that needs to know about your filesystem.
cfg.data_root = '/path/to/your/data/';      % where uigetdir starts
cfg.python_bin = '/path/to/env/bin/python';   % interpreter for the solver
end
