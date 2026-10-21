
function [R] = T_to_R (T, di)

if nargin<2 || isempty(di); di=1;end

    if di ==1
        R = Rt(T(1:3)) * Rx(T(6)) * Ry(T(5)) * Rz(T(4)); 
    elseif di ==0
        R = Rz(-T(4)) * Ry(-T(5))*Rx(-T(6)) * Rt(-T(1:3));
    else
        disp('T_to_R:: This direction does not exist. Termintaed with neutral transformation matrix.')
        R = eye(4); 
        return
    end

end






