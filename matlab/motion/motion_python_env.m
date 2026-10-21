function e = motion_python_env()
%MOTION_PYTHON_ENV  Interpreter and the library dirs the solver needs,
%                   all derived from one configured path.
%
%   e.bin        the python executable (motion_cfg_get('python_bin', ...))
%   e.libstdcxx  <env>/lib/libstdc++.so.6, '' if absent
%   e.nvidia     cusolver:cublas lib dirs from the env's site-packages, '' if absent
e.bin = motion_cfg_get('python_bin', 'python');
root  = fileparts(fileparts(e.bin));          % <env>

e.libstdcxx = fullfile(root,'lib','libstdc++.so.6');
if ~exist(e.libstdcxx,'file'), e.libstdcxx = ''; end

e.nvidia = '';
% dir() only globs the LAST component, so resolve python*/ first
pyd = dir(fullfile(root,'lib','python*'));
pyd = pyd([pyd.isdir]);
for k = 1:numel(pyd)
    base = fullfile(pyd(k).folder, pyd(k).name, 'site-packages', 'nvidia');
    if ~exist(base,'dir'), continue; end
    parts = {};
    for c = {'cusolver','cublas'}
        q = fullfile(base, c{1}, 'lib');
        if exist(q,'dir'), parts{end+1} = q; end %#ok<AGROW>
    end
    if ~isempty(parts), e.nvidia = strjoin(parts, ':'); break; end
end
end
