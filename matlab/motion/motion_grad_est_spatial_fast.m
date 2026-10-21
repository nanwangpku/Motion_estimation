function [est_out] = motion_grad_est_spatial_fast(motion_str, timing_str, init_str)
% MOTION_GRAD_EST_SPATIAL_FAST  faster drop-in for motion_grad_est_spatial.
%
% Same inputs, same outputs (est_out.m_est / .cr_est_poly / .c_est /
% .nt_sel_dB0 / .nt_sel_mo). motion_grad_est_spatial.m is left untouched.
%
% Profiled on 20260914_motion/5c350 (576 bins, 64 coils, 60x54x60, A6000):
%   original   loop1 1.05 s/bin + loop2 1.64 s/bin  ->  26 min
%     loop1: fft3c 36%, k-space staging 29%, motion_ops 17%
%     loop2: fft3c 52%, k-space staging 17%, gradient build 15%, flip 8%
%
% What changed, biggest first:
%
% 1. Only the sampled (ky,kz) lines x kx_sel are ever read back from each
%    transform -- about 0.4% of its output -- yet a full 3D FFT over
%    (Ny,Nz,Nx,Ncoils,Nt) was computed and then logically masked. F and the
%    mask now collapse into a partial DFT evaluated straight at those
%    points: contract (y,z) onto the sampled lines first, then x. The 1-D
%    matrices are built by pushing the identity through Funcs/fft/fftc, so
%    the transform and its normalisation match fft3c exactly.
%
% 2. In the motion loop fft3c(img_ref_SEs) sat inside the 6-direction
%    gradient loop, so 13 full 3D FFTs ran per bin where 7 were needed.
%
% 3. The dense (Ny,Nz,Nx,Ncoils,Nt) k-space buffer -- ~100 MB allocated and
%    zeroed per bin -- existed only to be masked down to a few hundred
%    samples. Sampled values are assembled directly now, and flip_flag is
%    folded into the sample indices instead of flipping the whole array
%    twice per bin.
%
% 4. img_grad was a CPU array written from gpuArrays, so every gradient
%    direction round-tripped GPU->CPU->GPU, and img_grad.*SEs_perm
%    materialised 1.2 GB. Sensitivities are now applied one direction and
%    one echo at a time on the GPU (~100 MB peak).
%
% 5. In the B0 loop motion_ops was called with all-zero motion parameters
%    every bin, which is an exact identity transform; skipped when the
%    parameters are zero.
%
% 6. pinv() on single columns and on a tall-thin matrix replaced by the
%    equivalent closed-form and backslash solves; the argmax-over-coils
%    voxel loop vectorised.
%
% 7. cr_est_poly came from pcg on the 9x9 normal equations, which hit its
%    default iteration cap every bin and left ~0.2-0.6% relative residual.
%    Replaced by an exact least-squares solve (one economy QR of the basis,
%    reused across bins). This is the one change that does NOT merely
%    reproduce the original to rounding -- it is more accurate.
%
% 8. The polynomial fit now uses Bas_norm (the [1 10 10 10 100 ...] scaled
%    basis the rest of the pipeline already uses) instead of Bas:
%    cond 7.6 vs 687, and 57 vs 4.7e5 on the normal equations. With the
%    exact solve above the fitted field is unchanged (reparametrisation
%    only, 4e-16), but cr_est_poly now comes out in the same units as the
%    per-shot stage. motion_str.grad_basis = 'raw' reverts.
%
% 9. The per-shot index prints are gone -- they wrote ~800 kB per run into
%    a SLURM log on NFS. Progress every 50 bins instead.
%
% Sample ordering: the original took samples in column-major order of the
% masked array, this one groups them by (echo, kx, point). Both are
% least-squares fits over the same SET of samples, so the estimates are
% unchanged -- only the row order of the intermediate matrices differs.

%% ---- unpack ------------------------------------------------------------
kspace_cart_orig_fid = motion_str.kspace_cart_orig_fid;
a_nav_refs  = motion_str.a_nav_refs;
Bas         = motion_str.Bas;
mask_B0_nav = motion_str.mask_B0_nav;
SEs_perm    = motion_str.SEs_perm;
TEs_fid     = motion_str.TEs_fid;
flip_flag   = motion_str.flip_flag;
size_nav    = motion_str.size_nav;
ky_fid      = motion_str.ky_fid;
kz_fid      = motion_str.kz_fid;
shot_fid    = motion_str.shot_fid;
ratios_all  = motion_str.ratios_all;

tmp = num2cell(size_nav);
[Ny_nav, Nz_nav, Nx_nav] = tmp{:};
Ncoils = size(SEs_perm,4);
clear tmp

fid_idx  = timing_str.fid_idx;
resp_bin = timing_str.resp_bin;
fid_int  = timing_str.fid_int;
nbins    = max(resp_bin);

if isfield(timing_str,'nt_sel_dB0') && ~isempty(timing_str.nt_sel_dB0)
    nt_sel_dB0 = timing_str.nt_sel_dB0;
else
    nt_sel_dB0 = [1 5];
end
if isfield(timing_str,'nt_sel_mo') && ~isempty(timing_str.nt_sel_mo)
    nt_sel_mo = timing_str.nt_sel_mo;
else
    nt_sel_mo = [1 3];
end
kx_sel = ceil(Nx_nav/5+1):ceil(Nx_nav/5*4);
nkx    = numel(kx_sel);

if isfield(init_str,'B0_est_spatial')
    B0_est_spatial = init_str.B0_est_spatial;
else
    B0_est_spatial = zeros([size_nav nbins]);
end

cls     = class(a_nav_refs);          % follow whatever precision came in
use_gpu = gpuDeviceCount > 0;

%% ---- centred 1-D DFT matrices, convention-matched to Funcs/fft/fftc ----
Ey_full = fftc(eye(Ny_nav,cls),1);    % (Ey*x)(k) == fftc(x,1)(k)
Ez_full = fftc(eye(Nz_nav,cls),1);
Ex_full = fftc(eye(Nx_nav,cls),1);
Ex      = Ex_full(kx_sel,:);          % (nkx, Nx)
if use_gpu
    SEs_perm = gpuArray(SEs_perm);
    Ex       = gpuArray(Ex);
end

m_est       = zeros(nbins,6);
cr_est_poly = zeros(nbins,9);
c_est       = zeros(nbins,Ncoils);
c0          = zeros(nbins,1);

%% ========================= loop 1: B0 ===================================
tic
for ii_bin = 1:nbins
    [sig, Eyz, te_vec] = gather_bin(ii_bin, nt_sel_dB0);
    if isempty(sig), continue; end

    img_ref = a_nav_refs(:,:,:,1,nt_sel_dB0) ...
        .* exp(1i*2*pi*B0_est_spatial(:,:,:,ii_bin).*reshape(TEs_fid(nt_sel_dB0),1,1,1,1,[]));
    if ii_bin > 1
        img_ref = img_ref .* exp(1i*c0(ii_bin-1)*reshape(TEs_fid(nt_sel_dB0),1,1,1,1,[]));
    end
    if any(m_est(ii_bin,:) ~= 0)   % all-zero params make motion_ops an identity
        [~,~,~,img_ref] = motion_ops(size_nav,m_est(ii_bin,:),[0 0 0],img_ref);
    end
    if use_gpu, img_ref = gpuArray(img_ref); end

    ref = sample_echoes(img_ref, SEs_perm, Eyz, Ex, nkx, Ncoils);   % (nsamp,Ncoils)

    C   = 1i*2*pi*ref.*te_vec;
    Ca  = [real(C); imag(C)];
    dka = [real(sig-ref); imag(sig-ref)];

    % pinv(v)*d for a column v is exactly (v'*d)/(v'*v)
    num = sum(Ca.*dka,1);
    den = sum(Ca.*Ca,1);
    c_est(ii_bin,:) = gather(num./den);
    c0(ii_bin)      = gather(sum(num)/sum(den));

    if mod(ii_bin,50)==0 || ii_bin==nbins
        fprintf('  [grad B0]     bin %d/%d  (%.1f s)\n', ii_bin, nbins, toc);
    end
end

%% ---- per-voxel B0 -> polynomial coefficients --------------------------
SEs_tmp = reshape(gather(SEs_perm),[],Ncoils);
[~,ii_c_max] = max(abs(SEs_tmp),[],2);        % was a loop over every voxel
B0s = reshape(c_est(:,ii_c_max).',[size_nav nbins]);

% Basis for the polynomial B0 fit. Bas' columns span three orders of
% magnitude (RMS 1 down to 0.0016), which is where the ill-conditioning
% came from: cond(Bas) = 687, and squaring it for the normal equations
% gave 4.7e5. Bas_norm -- the same basis with the [1 10 10 10 100 ...]
% scaling the rest of the pipeline already uses -- has cond 7.6 / 57.
% Under an exact solve the fitted field is identical either way (it is
% just a reparametrisation, verified to 4e-16); the normalised basis is
% the conditioned one, and it puts cr_est_poly in the same units as the
% per-shot stage, which is handed Basn_tmp.
if isfield(motion_str,'Bas_norm') && ~isempty(motion_str.Bas_norm) && ...
        ~(isfield(motion_str,'grad_basis') && strcmpi(motion_str.grad_basis,'raw'))
    Bas_fit = motion_str.Bas_norm;
else
    Bas_fit = Bas;
end
tmp2 = reshape(Bas_fit,[],9);
tmp2 = double(tmp2(mask_B0_nav(:),:));
% The original built the 9x9 normal equations and handed them to pcg, which
% stops at its default iteration cap (min(n,20) = 9) still ~0.2-0.6% away in
% relative residual -- so cr_est_poly carried that error. Solve the
% least-squares problem directly instead, and on the tall matrix rather than
% the normal equations, whose condition number is the square. tmp2 does not
% change with the bin, so the factorisation is computed once.
[Qb,Rb] = qr(tmp2,0);
for ii_bin = 1:nbins
    tmp1 = B0s(:,:,:,ii_bin);
    cr_est_poly(ii_bin,:) = (Rb \ (Qb' * double(tmp1(mask_B0_nav(:))))).';
end
B0_est_spatial = reshape(reshape(Bas_fit,[],9)*cr_est_poly.',[size_nav,nbins]);

%% ======================= loop 2: motion =================================
motion_grad = 0.2*eye(6);
tic
for ii_bin = 1:nbins
    [sig, Eyz, ~] = gather_bin(ii_bin, nt_sel_mo);
    if isempty(sig), continue; end
    sig = sig(:);

    ib = max(ii_bin-1,1);
    img_ref = a_nav_refs(:,:,:,:,nt_sel_mo) ...
        .* exp(1i*B0_est_spatial(:,:,:,ib).*reshape(TEs_fid(nt_sel_mo),1,1,1,1,[]));
    if use_gpu, img_ref = gpuArray(img_ref); end
    if ii_bin > 1
        [~,~,~,img_ref] = motion_ops(size_nav,m_est(ii_bin-1,:),[0 0 0],img_ref);
    end

    % hoisted: the reference transform is shared by all 6 directions
    ref = reshape(sample_echoes(img_ref, SEs_perm, Eyz, Ex, nkx, Ncoils), [], 1);

    M = zeros(numel(ref),6,cls);
    if use_gpu, M = gpuArray(M); end
    for ii = 1:6
        [~,~,~,img_d] = motion_ops(size_nav,motion_grad(ii,:),[0 0 0],img_ref);
        M(:,ii) = reshape(sample_echoes(img_d, SEs_perm, Eyz, Ex, nkx, Ncoils), [], 1) - ref;
    end
    M = M/0.2;

    Ma  = gather([real(M);   imag(M)]);
    dk  = sig - ref;
    dka = gather([real(dk);  imag(dk)]);
    step = (Ma \ dka).';                      % least squares, == pinv() here

    if ii_bin == 1
        m_est(ii_bin,:) = step;
    else
        m_est(ii_bin,:) = m_est(ii_bin-1,:) + step;
    end

    if mod(ii_bin,50)==0 || ii_bin==nbins
        fprintf('  [grad motion] bin %d/%d  (%.1f s)\n', ii_bin, nbins, toc);
    end
end

est_out.m_est       = m_est;
est_out.cr_est_poly = cr_est_poly;
est_out.c_est       = c_est;
est_out.nt_sel_dB0  = nt_sel_dB0;
est_out.nt_sel_mo   = nt_sel_mo;

%% ----------------------------------------------------------------------
    function [sig, Eyz, te_vec] = gather_bin(ii_bin, nt_list)
    % This bin's measured samples, plus the (y,z) DFT rows for the lines it
    % sampled. Replaces building, flipping and masking a dense array.
        Nt_l    = numel(nt_list);
        ky_s    = ky_fid(:,nt_list);
        kz_s    = kz_fid(:,nt_list);
        ii_shot = find(resp_bin == ii_bin & fid_int);
        if isempty(ii_shot), sig = []; Eyz = []; te_vec = []; return; end

        blocks = cell(Nt_l,1); Eyz = cell(Nt_l,1); tev = cell(Nt_l,1);
        for it = 1:Nt_l
            ky_v = ky_s(fid_idx(ii_shot),it);
            kz_v = kz_s(fid_idx(ii_shot),it);
            % a dense write let a later shot overwrite an earlier one on the
            % same line and counted it once; keep the last, to match
            [~, keep] = unique([ky_v(:) kz_v(:)], 'rows', 'last');
            keep = sort(keep);
            ky_v = ky_v(keep); kz_v = kz_v(keep); sh = ii_shot(keep);
            npts = numel(sh);

            datm = zeros(npts, nkx, Ncoils, cls);
            for jj = 1:npts
                ii_all = sh(jj);
                ii_out = floor((ii_all-1)/shot_fid)+1;
                ii_in  = mod(ii_all-1,shot_fid)+1;
                blk = fftc(permute(kspace_cart_orig_fid{ii_out}{ii_in}(:,nt_list(it),:),[1 3 2]),1) ...
                      .* ratios_all(ii_all);                 % (Nx, Ncoils)
                datm(jj,:,:) = blk(kx_sel,:);
            end
            blocks{it} = reshape(datm, npts*nkx, Ncoils);

            % flip_flag flipped the assembled array along dims 1 and 2,
            % which simply moves each sample to (Ny+1-ky, Nz+1-kz)
            if flip_flag
                ky_u = Ny_nav + 1 - ky_v;  kz_u = Nz_nav + 1 - kz_v;
            else
                ky_u = ky_v;               kz_u = kz_v;
            end
            E = zeros(npts, Ny_nav*Nz_nav, cls);
            for jj = 1:npts
                E(jj,:) = reshape(Ey_full(ky_u(jj),:).' * Ez_full(kz_u(jj),:), 1, []);
            end
            if use_gpu, E = gpuArray(E); end
            Eyz{it} = E;
            tev{it} = repmat(cast(TEs_fid(nt_list(it)),cls), npts*nkx, 1);
        end
        sig    = cat(1, blocks{:});
        te_vec = cat(1, tev{:});
        if use_gpu, sig = gpuArray(sig); te_vec = gpuArray(te_vec); end
    end
end

%% --------------------------------------------------------------------- %%
function out = sample_echoes(img, SEs, Eyz, Ex, nkx, Ncoils)
% Evaluate fft3c(img.*SEs) only at the sampled points.
%   img  : (Ny,Nz,Nx,1,Nt)      SEs : (Ny,Nz,Nx,Ncoils)
%   Eyz  : 1xNt cell, Eyz{t} is (npts_t, Ny*Nz)
%   out  : (sum_t npts_t*nkx, Ncoils), ordered point-fastest then kx
sz = size(img);
Ny = sz(1); Nz = sz(2); Nx = sz(3);
Nt = numel(Eyz);
out = cell(Nt,1);
for t = 1:Nt
    w = img(:,:,:,1,t) .* SEs;                       % (Ny,Nz,Nx,Ncoils)
    A = reshape(w, Ny*Nz, Nx*Ncoils);                % free reshape
    P = Eyz{t} * A;                                  % contract y,z first
    npts = size(P,1);
    P = reshape(permute(reshape(P, npts, Nx, Ncoils), [2 1 3]), Nx, npts*Ncoils);
    P = Ex * P;                                      % then x
    out{t} = reshape(permute(reshape(P, nkx, npts, Ncoils), [2 1 3]), npts*nkx, Ncoils);
end
out = cat(1, out{:});
end
