function [x,out] = l1_nf_sadmm(regType,A,y,lamda,p,ytrue,x0,rho,max_iter)
% l1_lp_admm_ac solves (with smoothing and linearization)
%
% minimize \lamda * ||e||_p^p + || x ||_{1, \epsilong}  s.t. e + Ax = y
% Inputs:
%	A: sensing matrix
%	y: CS data
%	lamda: regularization parameter
%	x0: initialization
%	xtrue: for debug, for calculation of errors
% Outputs
%	x: the CS recovery
%	out.e: the error with respect to the true
%	out.et: time index

[m,n]=size(A);
if nargin<9
    max_iter = 1000;
end
if nargin<8
    rho = 1e-3;  %  1e{-5, -4, ... ..., 1, 2, 3} % affect the prformance
end 
if nargin<7
    x = zeros(n,1);
else
    x = x0;
end

A2 = A'*A;
% eivalue = eig(A2);
% L = eig(A2);
L2 = 1000;
%e = zeros(m,1); w = zeros(m,1);
e = ones(m,1);  w = ones(m,1); % % tunable
tol = 1e-6;
out.xchag = [ ];
out.echag = [ ];
out.rechag = [ ];
out.obj = [];
%
wp1 = w;
ep = 1e-3; % \epsilong

% 1
% iA = (eye(n)-rho/(rho+L2)*A2)/L2;   % 2
% 1-used for non-orthonormal A, i.e., A*A' ~= I
% 2-used for orthonormal A, i.e., A*A' = I

for i = 1 : max_iter

    xm1 = x;
    em1 = e;
    %L2 = L2/1.05; % balance the prformance and the efficiency
    % e-step
    te = y-A*x-wp1/rho;
    switch(regType)
        case 1 % Lp
            e = shrinkage_Lp(te, p, 1/lamda, rho);
        case 2 % mcp
            e = shrinkage_Mcp(te, p, 1/lamda, rho);
        case 3 % scad
            e = shrinkage_Scad(te, p, 1/lamda, rho);
        otherwise
            assert(false);
    end

    % x-step
    iA = L2*eye(n) + rho*A2;
    z = L2*x + rho*(A'*(y-e-wp1/rho)) - x./sqrt(x.^2+ep*ep);
    x = iA\z;

    % w-step
    Ax = A*x;
    wp1 = wp1 + rho*(Ax - y + e);
    rho = rho*1.1;

    out.xchag = [out.xchag norm(x-xm1)];
    out.echag = [out.echag norm(e-em1)];
    out.rechag = [out.rechag norm(Ax-ytrue)];

    % terminate residuals
    if (norm(rho*(e-em1))< sqrt(n)*tol && norm(Ax-y+e)< sqrt(n)*tol)
        break;
    end

end



