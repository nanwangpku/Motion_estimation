function y = hamming_flat(N,L)
% this function create a hamming window with flat top
% N: total length of window. 
% L: true length of hamming window
% N-L, flat top
a = hamming(L);
y = zeros(N,1);
y(1:floor(L/2)) = a(1:floor(L/2));
y(floor(L/2)+1:floor(L/2)+N-L) = 1;
y(floor(L/2)+N-L+1:end) = a(floor(L/2)+1:end);

return
