function [x,out] = nuclear_nf_sadmm(A,Y,lamda,p,ytrue,regType,rho,x0,max_iter)
% l1_lp_admm_ac solves (with smoothing and acceleration)
%
%   minimize \lamda * ||E||_p^p + || X ||_{*, \epsilong}  s.t. E + AX = Y
% Inputs:
%	A: sensing matrix A=X
%	Y: CS data Y = X
%	lamda: regularization parameter
%	x0: initialization
%	xtrue: for debug, for calculation of errors
% Outputs
%	x: the CS recovery
%	out.e: the error with respect to the true
%	out.et: time index

%regType = 1;
%Convergence setup
if nargin < 9
    max_iter = 1000;
end
[m,n]=size(A);

if nargin<8
    rho = 1e-2; % 1e{-5, -4, ... ..., 1, 2, 3} % affect the prformance
end

%Initialize
if nargin<7
    x = eye(n,n);
else
    [x]= x0;
end

%e = zeros(m,n); w = zeros(m,n);
e = ones(m,n); w =  ones(m,n);
ABSTOL = 1e-6;
out.xchag = []; 
out.echag = []; 
out.obj = [];

tic;
wp1 = w;
ep = 1e-3; %\epsilong
L2 = 1000; % tunable
A2 = A'*A;
    % used for non-orthonormal A, i.e., A*A' ~= I
for i = 1 : max_iter
    xm1 = x;
    vm1 = e;
    L2 = L2/1.05; % balance the prformance and the efficiency
    %e-step
    te = Y-A*x-wp1/rho;
    e = solve_nonconvexl2(te,lamda,rho,p,regType);
    
    %x-step
    iA = inv(L2*eye(n) + rho*A2); 
    z = L2*x + rho*(A'*(Y-e-wp1/rho)) - x*(x'*x+ep*eye(size(x,2)))^(-0.5);
    x = iA*z;
    
    %w-step
    Ax = A*x;
    wp1 = wp1 + rho*(Ax - Y + e);
    rho = rho*1.1;
 
    out.xchag = [out.xchag norm(x-xm1)];
    out.echag  = [out.echag norm(Ax-ytrue)];
    %terminate when both primal and dual residuals are small
    if (norm(rho*(e-vm1))/norm(Y)<  ABSTOL && norm(Ax-Y+e)/norm(Y)< ABSTOL)
        break;
    end
end

%%
function [E] = solve_nonconvexl2(W,lamda,rho,p,regType)
n = size(W,2);
E = W;
for i = 1:n
    switch(regType)                                          % lamda and rho are both tuneable
        case 1 % lp
            eenum = shrinkage_Lp(norm(W(:,i)), p, 1/lamda, rho);
            E(:,i) = max(eenum,0)*W(:,i)/norm(W(:,i));
        case 2 % MCP
            eenum = shrinkage_Mcp(norm(W(:,i)), p, 1/lamda, rho);
            E(:,i) = max(eenum,0)*W(:,i)/norm(W(:,i));
        case 3 % SCAD
            eenum = shrinkage_Scad(norm(W(:,i)), p, 1/lamda, rho);
            E(:,i) = max(eenum,0)*W(:,i)/norm(W(:,i));
        otherwise
            assert(false);
    end
end
