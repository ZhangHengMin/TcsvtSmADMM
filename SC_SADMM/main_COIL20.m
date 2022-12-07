clear all;
close all;


addpath('Dataset');
addpath('Myfuns'); 
%
nCluster = 10;
load('.\Dataset\COIL20.mat');
Xo = fea';
gnd = gnd';
fprintf('nCluster = %.0f\n', nCluster);
class_sample_num = 100;  %
nums = nCluster * class_sample_num;    % number of data used for subspace segmentation
Xo = Xo(:,1:nums);
gnd = gnd(:,1:nums);
K = max(gnd);

X = Xo;
for ii = 1 : size(X,2)
    X(:,ii) = X(:,ii) /norm(X(:,ii)) ;
end

acc = 0; nmi00 = 0; time = 0;

%%
paras = [0.01 0.1, 0.5, 1.0, 3.0, 5.0 8.0 10];
p = [0.5 1.5 2.5]; % tunable
fun = {'lp', 'MCP', 'SCAD'};

for funtype = 1 : 3
    disp(['type_fun = ' num2str(fun{funtype})]);

    for j = 1 : length(paras)
        %
        fprintf(' lambda  = %.4f ', paras(j));
        tic;
        %
        [Z, output] = nuclear_nf_sadmm(X,X,paras(j),p(funtype),X,funtype);
        %
        time(j) = toc;
        %%

        acc(j) = run_clustering(Z, gnd);
        disp([' seg_acc=' num2str(acc(j)) ', cost_time=' num2str(time(j))] );
    end

end

