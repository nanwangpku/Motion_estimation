
function whichdev = choose_GPU()
clear am


    for j=1:gpuDeviceCount
        try
            gpudev=gpuDevice(j);
            am(j)=gpudev.AvailableMemory./gpudev.TotalMemory;
        catch
            am(j) = 0;
        end
    end
    [~,whichdev]=max(am)
    % gpudev = gpuDevice(whichdev);
return