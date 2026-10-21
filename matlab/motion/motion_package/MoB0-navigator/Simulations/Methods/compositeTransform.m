
function [Tcomp, mae] = compositeTransform(T, di)

%COMPOSITETRANSFORM computes the composite transformation parameters coming
% from a sequential application of transformations. Transformation
% convention follows sincRigidTransformation.m 
%   [TCOMP, MAE]=COMPOSITETRANSFORM(T,{DI})
%   * T contains the different transformations and associated parameters in the 5th dimension
%   * {DI} determines the direction of each of the transformations. Defaults to 1 (forward).
%   ** TCOMP is the composite transform (3 translations and 3 rotations)
%   ** MAE is the MAE of the rotation matrix calculated in both was. Used for error handling.
%

NT = size(T);
NTransforms = NT(5);

if nargin <2 || isempty(di); di = ones( [1 NTransforms]);end %Remove parameter dimension

%%%Calculte composite transformation matrix
Rcomp = eye(4);
for tIdx = 1:NTransforms
    Rcomp = T_to_R (dynInd(T,tIdx,5) , di(tIdx))  * Rcomp;
end

%%% Convert to Euler angles in the specified order
Tcomp = convertT(Rcomp, 'R2T', 1:3);

%%% test if resulting rotation matrix is close to iriginal
RcompTest = T_to_R(Tcomp,1);
mae = multDimMea(abs(Rcomp - RcompTest));
tol=1e-5;
if mae>tol;fprintf('compositeTransform: MAE of compositeTransform conversion bigger than %d - this error might accumulate throughout algorithm\n',tol);end

