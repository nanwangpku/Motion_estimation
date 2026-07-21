function [et, T, Th,recon_targ] = motion_ops(size_img, motion_params,pad_num, recon_mov)
% size_img should be in dim [Ny, Nz, Nx];
dims = numel(size_img);
size_img0 = size_img;
if sum(pad_num)>0
    if numel(pad_num) == 1
        pad_num = pad_num*ones(1,dims);  % this means for all dimension, pad the same amount of 0
    end
    size_img = size_img+pad_num*2;  % how many dimension, should be 3
    pad_for = @(x)padarray(x,pad_num);
    pad_adj = @(x)x(pad_num(1)+(1:size_img0(1)),pad_num(2)+(1:size_img0(2)),pad_num(3)+(1:size_img0(3)),:);

else
    pad_for = @(x)x;
    pad_adj = @(x)x;
end

dims = numel(size_img);  % how many dimension, should be 3
kGrid = cell(1,3); % kspace grid
rGrid = cell(1,3);  % image space grid
rkGrid = cell(2,3); % kspace grid * image space grid

[Z,Y,X] = meshgrid([-size_img(2)/2:size_img(2)/2-1],[-size_img(1)/2:size_img(1)/2-1],[-size_img(3)/2:size_img(3)/2-1]);
kGrid{1} = Y*2*pi/size_img(1);
kGrid{2} = Z*2*pi/size_img(2);
kGrid{3} = X*2*pi/size_img(3);

rGrid{1} = Y;
rGrid{2} = Z;
rGrid{3} = X;


per=[1 3 2;
    2 1 3];
for n=1:2
    for m=1:3
        rkGrid{n}{m}=rGrid{per(3-n,m)}.*kGrid{per(n,m)};
    end
end

%% generate matrix
% motion_params = load(temp_motion);  % motion path is the path of afni recon
pos_rot=motion_params(1:3)/180*pi;
pos_tra=motion_params([5 6 4]); % Ny(front->back), Nz(left->right), Nx (head->foot);
pos_tra=pos_tra.*[-1 -1 1];
pos_tra = pos_tra.*size_img./size_img0;  % this is to scale up for padding


% pos_tra = [0.6155    3.6154   -0.9594];
% pos_rot = [0.3690    0.0183   -0.0259];

et{1} = exp(-1i* (kGrid{1}*pos_tra(1) + kGrid{2}*pos_tra(2) + kGrid{3}*pos_tra(3)));

eth{1} = exp(1i* (kGrid{1}*pos_tra(1) + kGrid{2}*pos_tra(2) + kGrid{3}*pos_tra(3)));

theta=wrapToPi(pos_rot);
theta=wrapToPiHalf(theta);
tantheta2=theta/2;
tantheta2=tan(tantheta2);

sintheta=sin(theta);

et{2}=cell(1,3);
et{3}=cell(1,3);
for m=1:3
    et{2}{m}=exp(1i*(tantheta2(m)*rkGrid{1}{m}));%Tan exponential
    et{3}{m}=exp(-1i*(sintheta(m)*rkGrid{2}{m}));%Sin exponential
end

%% build operator
% T = @(x)pad_for(x);
% e_corner = exp(-1i* (kGrid{1}*(size_img(1)/2+1) + kGrid{2}*(size_img(2)/2+1) + kGrid{3}*(size_img(3)/2+1)));
T=@(x)fftc(fft2c(x),3);
% T=@(x)T(x).*M;
T=@(x)pad_for(T(x));

T=@(x)ifftc(ifft2c(T(x)),3);

for m=1:3  %Axis: 3-2-1: for LR(fast PE)-AP(slow PE)-FH(read) it would be yaw-roll-pitch
%     if any(et{5}{m}(:)==1);x=bsxfun(@times,x,1-et{5}{m})+bsxfun(@times,flipping(x,et{4}{m}),et{5}{m});end%FLIPPING FOR LARGER THAN 90DEG ROTATIONS
    T=@(x)fftc(T(x),per(1,m));
    T=@(x)T(x).*et{2}{m};
    T=@(x)ifftc(T(x),per(1,m));
    T=@(x)fftc(T(x),per(2,m));
    T=@(x)T(x).*et{3}{m};
    T=@(x)ifftc(T(x),per(2,m));
    T=@(x)fftc(T(x),per(1,m));
    T=@(x)T(x).*et{2}{m};
    T=@(x)ifftc(T(x),per(1,m));
end
T=@(x)fftc(fft2c(T(x)),3);
% T=@(x)T(x).*conj(e_corner);
T=@(x)T(x).*et{1}; % NW, test, 20240819
T=@(x)pad_adj(T(x));
T=@(x)ifftc(ifft2c(T(x)),3);

%% build hermitian operator
% e_corner = exp(-1i* (kGrid{1}*size_img(1)/2 + kGrid{2}*size_img(2)/2 + kGrid{3}*size_img(3)/2));
Th=@(x)fftc(fft2c(x),3);
Th=@(x)pad_for(Th(x));
Th=@(x)Th(x).*conj(et{1});
Th=@(x)ifftc(ifft2c(Th(x)),3);
for m=3:-1:1 %Axis: 3-2-1: for LR(fast PE)-AP(slow PE)-FH(read) it would be yaw-roll-pitch
%     if any(et{5}{m}(:)==1);x=bsxfun(@times,x,1-et{5}{m})+bsxfun(@times,flipping(x,et{4}{m}),et{5}{m});end%FLIPPING FOR LARGER THAN 90DEG ROTATIONS
    Th=@(x)fftc(Th(x),per(1,m));
    Th=@(x)Th(x).*conj(et{2}{m});
    Th=@(x)ifftc(Th(x),per(1,m));
    Th=@(x)fftc(Th(x),per(2,m));
    Th=@(x)Th(x).*conj(et{3}{m});
    Th=@(x)ifftc(Th(x),per(2,m));
    Th=@(x)fftc(Th(x),per(1,m));
    Th=@(x)Th(x).*conj(et{2}{m});
    Th=@(x)ifftc(Th(x),per(1,m));
end
Th=@(x)fftc(fft2c(Th(x)),3);
% Th=@(x)Th(x).*M;
Th=@(x)pad_adj(Th(x));
Th=@(x)ifftc(ifft2c(Th(x)),3);

%% do correction
if nargin>3
    
    recon_targ = T(recon_mov);
else
    recon_targ = zeros(size_img);
end
return