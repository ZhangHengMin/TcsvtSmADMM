function x = shrinkage_Mcp(b, p, lam, L)

r = lam/L;
n = length(b);
for j = 1 : n
    a = b(j);
    % p > 1
    if  abs(a) <= r
        xn = 0;
        
    elseif  abs(a) > p*r
        xn = a;
        
    else
        xn = sign(a)*p*(abs(a)-r)/(p-1);
    end
    x(j,:) = xn;
end

