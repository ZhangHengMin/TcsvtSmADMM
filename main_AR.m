close all;
clear all;


addpath('AR_Gray');
addpath('Myfuns');

% Image size and dim
Image_row_NUM = 50;
Image_column_NUM = 40;
NN = Image_row_NUM*Image_column_NUM;

% Image number
Class_Sample_NUM = 26;
Class_NUM = 120;
Total_Sample_NUM = Class_NUM*Class_Sample_NUM;

% total data
Choice_set = 1:26;
DAT = zeros(NN,Class_Sample_NUM,Class_NUM);

for r = 1:Class_NUM
    for t = 1:Class_Sample_NUM
        if r < 10
            string = ['.\AR_Gray\AR' int2str(0) int2str(0) int2str(r) '-' int2str(t) '.tif'];
        elseif r < 100
            string = ['.\AR_Gray\AR' int2str(0) int2str(r) '-' int2str(t) '.tif'];
        else
            string = ['.\AR_Gray\AR' int2str(r) '-' int2str(t) '.tif'];
        end
        A = imread(string,'tif');
        A = im2double(A);
        %A = A/norm(A,'fro');
        DAT(:,t,r) = A(:);
    end
end

% training data
Select_Class_Sample_Num = 7;
Select_total_Num = 2*Select_Class_Sample_Num;
AA_0 = [DAT(:,1:Select_Class_Sample_Num,2) DAT(:,1:Select_Class_Sample_Num,5) ];
AA = reshape(AA_0, [Image_row_NUM,Image_column_NUM, Select_total_Num]);
Train_DAT = reshape(AA,[NN,Select_total_Num]);
Train_NUM = Select_total_Num;

% test image
clear_B = reshape(DAT(:,1,2),[Image_row_NUM,Image_column_NUM]);
pp = 0.2; % noise level
B = imnoise(clear_B,'gaussian',pp);


lambda = [0.01]; % tunable
rr = [0.5 5 10]; % function parameter

%
for j = 1: length(lambda)
    fprintf('lambda=%7.4f\n',lambda(j));

    % nonfun Lp
    t0 = clock;
    [x_Lp,output_Lp] = l1_nf_sadmm(1,Train_DAT,B(:),lambda(j),rr(1),clear_B(:));
    time_Lp = etime(clock,t0);
    AxLp = reshape(Train_DAT*x_Lp,[50,40]);
    ReER_Lp= norm(AxLp-clear_B,'fro')/norm(clear_B,'fro');
    Class_Recon_LpError=zeros(2,1);
    for t = 1 : 2
        X = Train_DAT(:,(t-1)*Select_Class_Sample_Num+1:t*Select_Class_Sample_Num);
        Class_W = x_Lp((t-1)*Select_Class_Sample_Num+1:t*Select_Class_Sample_Num);
        Recon_Train_Sample = X*Class_W;
        Reconsruction_error  = AxLp(:)-Recon_Train_Sample;
        Class_Recon_LpError(t) = sum(abs(Reconsruction_error));
    end
    fprintf('cost_time_Lp=%7.4f, value_rerr_Lp=%7.5f\n', time_Lp, ReER_Lp);

    % nonfun Mcp
    t0 = clock;
    [x_Mcp,output_Mcp] = l1_nf_sadmm(2,Train_DAT,B(:),lambda(j),rr(2),clear_B(:));
    time_Mcp = etime(clock,t0);
    AxMcp = reshape(Train_DAT*x_Mcp,[50,40]);
    ReER_Mcp= norm(AxMcp-clear_B,'fro')/norm(clear_B,'fro');
    Class_Recon_McpError=zeros(2,1);
    for t = 1 : 2
        X = Train_DAT(:,(t-1)*Select_Class_Sample_Num+1:t*Select_Class_Sample_Num);
        Class_W = x_Mcp((t-1)*Select_Class_Sample_Num+1:t*Select_Class_Sample_Num);
        Recon_Train_Sample = X*Class_W;
        Reconsruction_error  = AxMcp(:)-Recon_Train_Sample;
        Class_Recon_McpError(t) = sum(abs(Reconsruction_error));
    end
    fprintf('cost_time_Mcp=%7.4f, value_rerr_Mcp=%7.5f\n', time_Mcp, ReER_Mcp);

    % nonfun Scad
    t0 = clock;
    [x_Scad,output_Scad] = l1_nf_sadmm(3,Train_DAT,B(:),lambda(j),rr(3),clear_B(:));
    time_Scad = etime(clock,t0);
    AxScad = reshape(Train_DAT*x_Scad,[50,40]);
    ReER_Scad= norm(AxScad-clear_B,'fro')/norm(clear_B,'fro');
    Class_Recon_ScadError=zeros(2,1);
    for t = 1 : 2
        X = Train_DAT(:,(t-1)*Select_Class_Sample_Num+1:t*Select_Class_Sample_Num);
        Class_W = x_Scad((t-1)*Select_Class_Sample_Num+1:t*Select_Class_Sample_Num);
        Recon_Train_Sample = X*Class_W;
        Reconsruction_error  = AxScad(:)-Recon_Train_Sample;
        Class_Recon_ScadError(t) = sum(abs(Reconsruction_error));
    end
    fprintf('cost_time_Scad=%7.4f, value_rerr_Scad=%7.5f\n', time_Scad, ReER_Scad);


end

% visiual results
figure(1);
subplot(3,1,1); imshow([clear_B B B-AxLp AxLp],'border','tight','initialmagnification',50); hold on;
subplot(3,1,2); imshow([clear_B B B-AxMcp AxMcp],'border','tight','initialmagnification',50); hold on;
subplot(3,1,3); imshow([clear_B B B-AxScad AxScad],'border','tight','initialmagnification',50); hold off;

 