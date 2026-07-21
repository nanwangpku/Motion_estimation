% computeSensePlusMotionDataConsistency.m
% NW, 20240806

% These scripts and data provide example code for the method described in:
% Polak et al. 2021, "Scout Accelerated Motion Estimation and Correction (SAMER)"


% This function evalutes the SENSE plus motion forward model for a given
% input image and motion parameters and computes the data consistency error
% with respect to the acquired/simualted k-space data


function [obj]  = computeSensePlusMotionDataNew( x, img, in_str, kspace_data, sens, mask_nav)
    
    if isfield(in_str,'size_nav')  % size nav is the spatial size of image, no temporal size
        size_nav = instr.size_nav;
    else
        size_nav = size(img); if numel(size_nav)>3; size_nav = size_nav(1:3); end
    end

    if isfield(in_str,'dphs')  % background phase
        dphs = in_str. dphs;
    else
        dphs = ones(size_nav);
    end

    Bas = in_str.Bas;
    bcoef = in_str.bcoef;
    TEs = in_str.TEs;
    mot_params = in_str.mot;
    pcoef = in_str.pcoef;

    % NW, disable 20240809
    % % pre-compute fft-scaling
    % fft_norm = sqrt(size(img,1)*size(img,2));
    phs_bg0 = exp(1i*reshape(sum(Bas.*pcoef,2),size_nav) );
    phs_B0 = exp(1i*2*pi*reshape(sum(Bas.*bcoef,2),size_nav).*reshape(TEs,1,1,1,1,[])); % B0 induced phase, size_nav+1+T, 1 is to allow the sensitivity
    img_tmp = img .* dphs .* phs_B0 .* phs_bg0;
    % perform translation and rotation 
    [~,~,~,img_warp] = motion_ops(size_nav, x(1:6),0,img_tmp);
    
    % Multiply with the coil sensitivity and Fourier transform
    forwardModelEval = fft3c(sens .*img_warp);

    % Mask the data for a given shot
    forwardModelEval_masked = mask_nav.* forwardModelEval;
    kspace_data_masked = mask_nav.* kspace_data ;  
    
    % compute the data consistency error
    data_consistency_error = norm(forwardModelEval_masked(:)-kspace_data_masked(:)) / norm(kspace_data_masked(:));
    
    % return the data consistency error
    obj = double(data_consistency_error);
     
end



