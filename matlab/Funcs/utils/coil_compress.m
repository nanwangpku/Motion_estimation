function [SEs_r, kspace_r] = coil_compress(SEs_perm, kspace_cart_orig_fid, Ncoils_target, method)
%COIL_COMPRESS  Reduce SEs_perm / kspace_cart_orig_fid to Ncoils_target channels.
%
% method: 'hard' (top energy-ranked physical coils, whole-FOV weighting) or
%         'svd' (SVD compression of the sensitivity maps into
%         Ncoils_target virtual coils, same matrix applied to SEs and
%         every shot's k-space data).
%
% If Ncoils_target >= the current channel count, returns the input
% unchanged (no-op).

Ncoils_full = size(SEs_perm, 4);

if Ncoils_target >= Ncoils_full
    SEs_r = SEs_perm;
    kspace_r = kspace_cart_orig_fid;
    return
end

if strcmpi(method, 'svd')
    sz4 = size(SEs_perm);
    SEs_mat = reshape(SEs_perm, [], Ncoils_full);
    [~, ~, V] = svd(SEs_mat, 'econ');
    Wcc = V(:, 1:Ncoils_target);
    SEs_r = reshape(SEs_mat * Wcc, sz4(1), sz4(2), sz4(3), Ncoils_target);

    kspace_r = kspace_cart_orig_fid;
    for ii = 1:size(kspace_r,2)
        for jj = 1:size(kspace_r{1},2)
            dat = kspace_r{ii}{jj};
            sz = size(dat);
            dat = reshape(dat, [], sz(end)) * Wcc;
            kspace_r{ii}{jj} = reshape(dat, [sz(1:end-1), Ncoils_target]);
        end
    end
else
    coil_weight = sqrt(sum(abs(reshape(SEs_perm,[],Ncoils_full)).^2, 1));
    [~, coil_order] = sort(coil_weight, 'descend');
    coil_sel = sort(coil_order(1:Ncoils_target));

    SEs_r = SEs_perm(:,:,:,coil_sel);
    kspace_r = kspace_cart_orig_fid;
    for ii = 1:size(kspace_r,2)
        for jj = 1:size(kspace_r{1},2)
            kspace_r{ii}{jj} = kspace_r{ii}{jj}(:,:,coil_sel);
        end
    end
end

end
