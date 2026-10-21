function [mask_si, info] = detect_and_mask_shoulders(vol_ref, si_dim, varargin)
%DETECT_AND_MASK_SHOULDERS  Heuristic shoulder detection + SI-direction mask.
%
%   Built for infant head scout/navigator volumes where the prescribed FOV
%   sometimes includes part of the shoulders/torso below the neck. Works
%   purely off image intensity - no atlas/registration required, only
%   base MATLAB + Image Processing Toolbox (bwareafilt) + Signal Processing
%   Toolbox (findpeaks).
%
%   Assumes a fixed SI convention: index 1 along si_dim is the "clean" side
%   (above the top of the head, background) and increasing index moves
%   toward the feet, i.e. any shoulder/torso content sits at the high-index
%   end. This matches this codebase's navigator orientation; flip the volume
%   first if your data runs the other way.
%
%   [mask_si, info] = detect_and_mask_shoulders(vol_ref, si_dim, 'Name', Value, ...)
%
% Inputs
%   vol_ref  - reference 3D volume (magnitude or complex) used to detect the
%              neck/shoulder boundary. Use the scout image (a_nav_refs) since
%              it typically has the best SNR of all navigator frames. If you
%              have multiple reps/bins, average abs(.) across them first for
%              a more robust profile (see example at the bottom of this file).
%   si_dim   - dimension index of vol_ref that runs head<->foot (SI). For
%              this codebase, a_nav_refs / a_nav_sense are sized
%              [Ny_nav, Nz_nav, Nx_nav, ...], so si_dim = 3.
%
% Name-Value options (all optional, defaults in parentheses)
%   'ThreshFrac'      (0.20) fraction of the whole volume's max magnitude
%                     used to binarize every slice before measuring its
%                     cross-sectional area. A single global threshold (not
%                     a per-slice one) keeps the area profile comparable in
%                     absolute terms across slices.
%   'SmoothSpan'      (5)    moving-average span (# slices) applied to the
%                     raw area-vs-slice profile before peak/valley search.
%   'EdgeSlices'      (3)    number of slices averaged at the start of the
%                     SI axis to check for a "clean start".
%   'EdgeThresh'      (0.30) normalized-profile level with two roles: (1)
%                     the first EdgeSlices must average below this for the
%                     start to count as "clean" (background above the
%                     head), which is what lets the first bulge found be
%                     trusted as the head; (2) a dip between two bulges
%                     must fall below this to count as a real neck, as
%                     opposed to head-only width fluctuation.
%   'MinPeakDistance' (5)    minimum slice separation between two profile
%                     peaks for both to be treated as distinct anatomical
%                     bulges (head vs. shoulder) rather than the same peak.
%   'NeckMargin'      (2)    number of extra slices, on the head side of
%                     the detected neck, to start the taper from. The
%                     detected neck index is the profile's minimum, but
%                     residual shoulder signal often extends a bit past
%                     it (partial volume / smoothing), so starting the
%                     ramp-down a couple of slices earlier ensures the
%                     neck itself is fully removed rather than just its
%                     narrowest point.
%   'TaperLen'        (5)    number of slices over which the mask ramps
%                     from 1 to 0 (raised-cosine) instead of a hard cutoff,
%                     to avoid a sharp step that could ring in later
%                     Fourier-domain steps.
%   'Verbose'         (false) if true, plots the profile with detected
%                     peaks/neck for visual QC.
%
% Outputs
%   mask_si  - Nsi x 1 vector, values in [0,1], along si_dim. Ones where
%              anatomy should be kept, tapering to zero past the detected
%              neck. All ones if no shoulders are detected.
%   info     - diagnostic struct:
%                .has_shoulder    logical
%                .neck_idx        detected neck (profile minimum) index
%                .mask_start_idx  index where the taper actually starts
%                                 (neck_idx - NeckMargin, clamped to 1)
%                .peak_locs       indices of detected profile bulges
%                .profile         raw per-slice area profile
%                .profile_smooth  smoothed profile
%
% Shape logic
%   A head-only scan produces a single-bulge profile: it rises from
%   background, peaks once over the head, and tapers back toward
%   background - it never dips below EdgeThresh and then rises again.
%   A scan that also contains the shoulders produces a two-bulge profile:
%   the head peak, a dip that reaches background level (< EdgeThresh, the
%   neck), and a second rise (the shoulders). Requiring both a real second
%   peak AND a dip below EdgeThresh between the two peaks is what
%   distinguishes true shoulders from ordinary head-shape variation.
%
% Example
%   scout_vol = mean(abs(a_nav_refs), 5);      % collapse scout reps -> 3D
%   [mask_si, info] = detect_and_mask_shoulders(scout_vol, 3, 'Verbose', true);
%   if info.has_shoulder
%       a_nav_refs  = apply_si_mask(a_nav_refs,  mask_si, 3);
%       a_nav_sense = apply_si_mask(a_nav_sense, mask_si, 3);
%   end
%
% See also: apply_si_mask

%% ---- parse options ----
p = inputParser;
addParameter(p, 'ThreshFrac',      0.20);
addParameter(p, 'SmoothSpan',      5);
addParameter(p, 'EdgeSlices',      3);
addParameter(p, 'EdgeThresh',      0.30);
addParameter(p, 'MinPeakDistance', 5);
addParameter(p, 'NeckMargin',      2);
addParameter(p, 'TaperLen',        5);
addParameter(p, 'Verbose',         false);
parse(p, varargin{:});
opt = p.Results;

%% ---- bring si_dim to position 3 so the rest of the code can index vol(:,:,k) ----
vol = abs(vol_ref); % magnitude only - phase is irrelevant for a shape/area profile
ndim_v = ndims(vol);
perm_order = 1:ndim_v;
perm_order([3 si_dim]) = perm_order([si_dim 3]);
vol = permute(vol, perm_order);
Nsi = size(vol, 3);

%% ---- 1. per-slice cross-sectional area profile (global-max threshold) ----
profile = compute_area_profile(vol, opt.ThreshFrac);

%% ---- 2. smooth to suppress single-slice noise before peak/valley search ----
profile_smooth = movmean(profile, opt.SmoothSpan);

%% ---- 3. normalize to [0,1] so thresholds are scale-free ----
prof_range = max(profile_smooth) - min(profile_smooth);
prof_norm  = (profile_smooth - min(profile_smooth)) / (prof_range + eps);

info = struct('has_shoulder', false, 'neck_idx', [], 'mask_start_idx', [], ...
    'peak_locs', [], 'profile', profile, 'profile_smooth', profile_smooth);

%% ---- 4. require a clean start: this is what lets us trust that the first ----
%%         bulge found below is genuinely the head, not a torso already ----
%%         filling the FOV at slice 1. ----
n_edge  = min(opt.EdgeSlices, Nsi);
edge_lo = mean(prof_norm(1:n_edge));
if edge_lo >= opt.EdgeThresh
    warning('detect_and_mask_shoulders:noCleanStart', ...
        'SI profile does not start at background; cannot reliably localize the head. Skipping masking.');
    mask_si = ones(Nsi, 1);
    if opt.Verbose, plot_profile(prof_norm, [], [], opt, []); end
    return
end

%% ---- 5. find distinct anatomical bulges in the profile shape. A head-only ----
%%         scan has exactly one; a head+shoulder scan has (at least) two. ----
[~, pk_locs] = findpeaks(prof_norm, 'MinPeakHeight', opt.EdgeThresh, ...
    'MinPeakDistance', opt.MinPeakDistance);

if numel(pk_locs) < 2
    % Single bulge -> head only, tapering back to background. No shoulders.
    mask_si = ones(Nsi, 1);
    info.peak_locs = pk_locs;
    if opt.Verbose, plot_profile(prof_norm, [], [], opt, pk_locs); end
    return
end

%% ---- 6. neck = deepest point between the head bulge (1st peak) and the ----
%%         shoulder bulge (2nd peak) ----
head_idx     = pk_locs(1);
shoulder_idx = pk_locs(2);
[valley_val, valley_idx_local] = min(prof_norm(head_idx:shoulder_idx));
neck_idx = valley_idx_local + head_idx - 1;

%% ---- 7. only accept a genuine neck if the dip truly reaches background; ----
%%         otherwise the two "bulges" are just width variation within one ----
%%         structure (e.g. an oblong head), not a real head/shoulder split ----
if valley_val >= opt.EdgeThresh
    warning('detect_and_mask_shoulders:noNeckFound', ...
        'Two bulges were found but the dip between them does not reach background; skipping masking.');
    mask_si = ones(Nsi, 1);
    info.peak_locs = pk_locs;
    if opt.Verbose, plot_profile(prof_norm, [], [], opt, pk_locs); end
    return
end

%% ---- 8. build a smooth (raised-cosine) taper mask, starting a bit before ----
%%         the detected neck so the neck itself is fully removed rather ----
%%         than just its narrowest slice ----
mask_start_idx = max(1, neck_idx - opt.NeckMargin);
mask_si = build_taper_mask(Nsi, mask_start_idx, opt.TaperLen);

info.has_shoulder   = true;
info.neck_idx       = neck_idx;
info.mask_start_idx = mask_start_idx;
info.peak_locs      = pk_locs;

if opt.Verbose
    plot_profile(prof_norm, neck_idx, mask_start_idx, opt, pk_locs);
end

end

%% ===================== local helper functions =====================

function profile = compute_area_profile(vol, thresh_frac)
% Cross-sectional area (voxel count of the single largest connected
% foreground component) for every slice along dimension 3 of vol.
% Using the largest component (not raw thresholded voxel count) discards
% isolated noise peaks / wrap-around artifacts that would otherwise
% masquerade as anatomy. The binarization threshold is a single fraction
% of the whole volume's max magnitude (not each slice's own max), so
% slices with little or no signal don't get their noise floor inflated
% into a spurious "area".
Nsi = size(vol, 3);
profile = zeros(Nsi, 1);
vol_max = max(vol(:));
thresh = thresh_frac * vol_max;
for k = 1:Nsi
    if vol_max <= 0
        continue
    end
    bw = vol(:, :, k) > thresh;
    bw = bwareafilt(bw, 1);
    profile(k) = nnz(bw);
end
end

function mask = build_taper_mask(Nsi, ramp_start_idx, taper_len)
% Raised-cosine ramp from 1 (kept) to 0 (masked out), starting at
% ramp_start_idx and finishing taper_len slices into the shoulder side
% (index increasing toward the feet, per this function's fixed SI
% convention).
mask = ones(Nsi, 1);
ramp_start = ramp_start_idx;
ramp_end   = min(Nsi, ramp_start_idx + taper_len - 1);
n = ramp_end - ramp_start + 1;
mask(ramp_start:ramp_end) = 0.5 * (1 + cos(pi * (0:n-1)' / max(1, n-1)));
mask(ramp_end+1:end) = 0;
end

function plot_profile(prof_norm, neck_idx, mask_start_idx, opt, pk_locs)
figure('Name', 'detect_and_mask_shoulders QC');
plot(prof_norm, 'LineWidth', 1.5); hold on
yline(opt.EdgeThresh, '--', 'EdgeThresh');
if ~isempty(pk_locs)
    plot(pk_locs, prof_norm(pk_locs), 'kv', 'MarkerFaceColor', 'k');
end
if ~isempty(neck_idx)
    xline(neck_idx, 'r-', sprintf('neck @ %d', neck_idx));
end
if ~isempty(mask_start_idx)
    xline(mask_start_idx, 'm--', sprintf('mask starts @ %d', mask_start_idx));
end
xlabel('SI slice index'); ylabel('normalized cross-sectional area');
title('Shoulder detection profile'); grid on
end
