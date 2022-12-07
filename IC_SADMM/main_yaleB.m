clear all;
close all;
clc;

%addpath(genpath('WSNR'));
addpath('Dataset');
addpath('Prox_fun');

Image_row_NUM   = 96;
Image_column_NUM  = 84;
NN = Image_row_NUM * Image_column_NUM;
fun = {'lp', 'MCP', 'SCAD'};

for regTypenn = 1 : 3
    disp(['type_fun = ' num2str(fun{regTypenn})]);


    Class_NUM       = 38;
    Class_Train_NUM = 7;
    Class_Test_NUM  = 14;
    Train_NUM        = Class_NUM * Class_Train_NUM;
    Test_NUM         = Class_NUM * Class_Test_NUM;
    load('.\Dataset\subset4_96_84.mat');

    %% reshape and nomalize data
    select_Class_Train_NUM = Class_Train_NUM;
    Train_DAT = reshape(Train_DAT, [NN select_Class_Train_NUM Class_NUM]);

    %
    Train_SET = zeros(NN, select_Class_Train_NUM, Class_NUM);
    for jj = 1 : Class_NUM
        for j = 1 : select_Class_Train_NUM
            temp = Train_DAT(:, j, jj);
            temp = temp / norm(temp, 'fro');
            Train_SET(:, j, jj) = temp;
        end
    end
    %Train_SET = reshape(Train_SET, [NN Class_Train_NUM Class_NUM]);
    Train_DAT = reshape(Train_SET,[Image_row_NUM, Image_column_NUM, select_Class_Train_NUM, Class_NUM]);
    Test_SET = zeros(NN,Test_NUM);
    for ii = 1 : Test_NUM
        temp = Test_DAT(:, ii);
        temp = temp / norm(temp, 'fro');
        Test_SET(:, ii) = temp;
    end
    Test_SET = reshape(Test_SET, [NN Class_Test_NUM Class_NUM]);
    Test_DAT = reshape(Test_SET, [Image_row_NUM, Image_column_NUM,Class_Test_NUM,Class_NUM]);

    %% Nuclear norm

    all_Paras = [0.001 0.01 0.05 0.1 0.5 1.0];


    for index_p = 1:length(all_Paras)
        Regress_Para = all_Paras(index_p);
        tic;
        [Miss_NUM_Nu, minErr] = Classifier_sadm_f(regTypenn, Train_DAT, Test_DAT, Regress_Para);
        time_cost = toc;
        Recognition_Rates = (Test_NUM-Miss_NUM_Nu)/Test_NUM;
        disp([' lambda = ' num2str(Regress_Para), ' Reco_Rates= ' num2str(Recognition_Rates), ' Timecost== ' num2str(time_cost)]);
    end

end







