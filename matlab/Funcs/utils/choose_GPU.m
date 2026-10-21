
function whichdev = choose_GPU()
clear am
    n = gpuDeviceCount;
    if n == 0
        error('choose_GPU:noGPU', ...
            ['no GPU visible to MATLAB. Run on a GPU node (or via srun ' ...
             '--gres=gpu:1); with opts.use_slurm = false you must already ' ...
             'be inside a GPU allocation.']);
    end
    am = zeros(1, n);


    for j=1:n
        try
            gpudev=gpuDevice(j);
            % absolute free memory, not the free FRACTION: a fraction prefers an
            % empty 24 GB A5000 over a 48 GB A6000 with a little in use, and the
            % 0.75-mm recon (peak ~46 GB) then runs out of memory on it
            am(j)=gpudev.AvailableMemory;
        catch
            am(j) = 0;
        end
    end
    [~,whichdev]=max(am)
    % gpudev = gpuDevice(whichdev);
return