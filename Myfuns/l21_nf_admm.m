function [x,out] = l21_nf_admm(regType, A, y, Class_NUM, lamda, inp, x0, rho, max_iter)
% l1_lp_admm_ac solves (with smoothing and linearization)
%
% minimize minimize \lamda * ||E||_Sp^p + || x ||_{21, \epsilong}  s.t. E + A(x) = Y
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
[p,q]=size(y);
class_sample = n/Class_NUM;
if nargin<9
    max_iter = 1000;
end
if nargin<8
    rho = 1e1; % 1e{-5, -4, ... ..., 1, 2, 3} % affect the prformance
end
if nargin<7
    x = zeros(n,1);
else
    x = x0;
end
if nargin<6
    if regType == 1
        inp = 0.5;
    elseif regType == 2
        inp = 1.5;
    else
        inp = 2.5;
    end
end
A2 = A'*A;
% eivalue = eig(A2);
% L2 = eig(A2);
%ee = zeros(p,q); w = zeros(p,q);
ee = ones(p,q);  w = ones(p,q);
tol = 1e-3;
out.xchag = [ ];
out.echag = [ ];
out.rechag = [ ];
out.obj = [];
Ax = reshape(A*x,[p,q]);
wp1 = w;
ep = 1e-3; % \epsilong
L2 = 1000;
% 1
%iA = (eye(n)-rho/(rho+L2)*A2)/L2;   % 2
% 1-used for non-orthonormal A, i.e., A*A' ~= I
% 2-used for orthonormal A, i.e., A*A' = I

for i = 1 : max_iter

    xml = x;
    eml = ee;
    %
    L2 = L2/1.05;  % balance the prformance and the efficiency
    % E-step
    tie = y-Ax-wp1/rho;
    [U, D, V] = svd(tie, 'econ');
    te = diag(D);
    switch(regType)
        case 1
            sve = shrinkage_Lp(te, inp, 1/lamda, rho);
        case 2 % mcp
            sve = shrinkage_Mcp(te, inp, 1/lamda, rho);
        case 3 % scad
            sve = shrinkage_Scad(te, inp, 1/lamda, rho);
        otherwise
            assert(false);
    end
    ee =U(:,1:size(sve))*diag(sve)*V(:,1:size(sve))';

    % x-step
    iA = L2*eye(n) + rho*A2;
    z = L2*x + rho*(A'*(y(:)-ee(:)-wp1(:)/rho)) - x./sqrt(x.^2+ep*ep);
    x = iA\z;

    % w-step
    Ax = reshape(A*x, [p,q]);
    wp1 = wp1 + rho*(Ax - y + ee);
    rho = rho*1.1;


    out.xchag = [out.xchag norm(x-xml)];
    out.echag = [out.echag norm(ee-eml)];

    % terminate residuals
    if (norm(ee-eml)< sqrt(n)*tol && norm(x-xml)< sqrt(n)*tol)
        break;
    end

end

out.AX = Ax(:);

