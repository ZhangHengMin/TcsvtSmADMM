function [Miss_NUM_Nu, Class_Reconstruction_Errors]= Classifier_sadm_f(regType, Train_DAT,Test_DAT,lambda)
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if nargin < 2
    error('Not enought arguments!');
end

% The information from input data
[Image_row_NUM, Image_column_NUM, Class_Train_NUM, Class_NUM] = size(Train_DAT);
NN = Image_row_NUM*Image_column_NUM;
Train_NUM = Class_NUM*Class_Train_NUM;
Train_DAT = reshape(Train_DAT,[Image_row_NUM, Image_column_NUM, Train_NUM]);
Train_DAT = reshape(Train_DAT,[NN,Train_NUM]);
[Image_row_NUM, Image_column_NUM, Class_Test_NUM, Test_Class_NUM] = size(Test_DAT);


Miss_NUM_Nu = 0;
% parfor km = 1 %: Test_Class_NUM*Class_Test_NUM
%     k = fix((km-1)/Class_Test_NUM)+1;
%     m = km - (k-1)*Class_Test_NUM;

for k = 1 : Test_Class_NUM % class number
    for m = 1 : Class_Test_NUM % test sample number of every class
        Test = Test_DAT(:,:,m,k); % determine test sample

        % the following are sub_function
        [W_all, output] = l21_nf_admm(regType, Train_DAT, Test, Class_NUM, lambda);

        % choose every class train_data
        Class_Reconstruction_Errors = zeros(Class_NUM,1);
        for t=1:Class_NUM
            % the meaning of following code
            X=Train_DAT(:,(t-1)*Class_Train_NUM+1:t*Class_Train_NUM);
            Class_W=W_all((t-1)*Class_Train_NUM+1:t*Class_Train_NUM);
            Reconstruction_Train_Sample= X*Class_W; % only use same class weight
            Reconsruction_error = (output.AX(:)-Reconstruction_Train_Sample)/sum(Class_W.*Class_W);

            % Nuclear Norm Classifier
            Differ_Mat=reshape(Reconsruction_error, [Image_row_NUM, Image_column_NUM]);
            Singular_Value_Vector = svd(Differ_Mat);
            Class_Reconstruction_Errors(t) = sum(abs(Singular_Value_Vector));

        end
        [~,Class_No_Nu] = min(Class_Reconstruction_Errors);
        if Class_No_Nu~=k % strncmp is to compare the first n characters of two strings
            Miss_NUM_Nu=Miss_NUM_Nu+1;
        end
        %
    end
    %
end

%matlabpool close

