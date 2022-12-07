function x = shrinkage_Scad(b, p, lam, L)

% p > 2
r = lam/L;
n = length(b);
for j = 1 : n
    a = b(j);
    % 
    if  abs(a) <= 2*r
        xn = sign(a)*max(0, abs(a)-r);
        
    elseif  abs(a) > 2*r && abs(a) <= p*r
        xn = (a*(p-1)-sign(a)*r*p)/(p-2);
        
    else
        xn = a;
    end
    x(j,:) = xn;
end