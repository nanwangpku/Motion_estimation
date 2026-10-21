

function [B, Bcoco] = createSkopeBasis(N,MS,order)

if nargin<2 || isempty(order); order = 5:16;end

dimStore=6;
perm = [1:3 4:dimStore];

Ntemp = N(perm(1:3));
[pGrid]=generateGrid(Ntemp,[],Ntemp,ceil((Ntemp+1)/2));
[RO,PE,SL] = ndgrid(pGrid{1}(:),pGrid{2}(:),pGrid{3}(:));

%Flip : this was confirmed that the Skope linear basis then corresponds to the DFT phase direction
X = flip(SL,3);
Y = flip(PE,2);
Z = flip(RO,1);

% X = flip(SL,3);
% Y = flip(PE,2);
% Z = flip(RO,1);

fprintf('Need to permute the x and z as we also did this in the recon pipeline.\n');
crds = cat(2,X(:),Y(:),Z(:)).*MS(:).'/1e3;%Im meters%TODO: MS needs to be permuted as well

B = [];
for i=order
    if i>16
        B = cat(2,B,kbase_coco(i-16,crds));
    else
        B = cat(2,B,kbase_spha(i,crds));
    end
end
Bcoco = [];
for i=1:4; Bcoco = cat(2,Bcoco,kbase_coco(i,crds));end

B = resSub(B,1,Ntemp);
Bcoco = resSub(Bcoco,1,Ntemp);

B = resPop(B,4,[],dimStore);
Bcoco = resPop(Bcoco,4,[],dimStore);

B = permute(B,perm);
Bcoco = permute(Bcoco,perm);