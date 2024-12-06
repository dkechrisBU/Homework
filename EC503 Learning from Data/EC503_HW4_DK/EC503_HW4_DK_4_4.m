%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ENG EC 503 (Ishwar) Fall 2023
% HW 4.4
% <Demetrios Kechris dkechris@bu.edu>
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% You are always welcome to vectorize your loop!

clear, clc, close all,
%% 4.4 a) Normalization of data

disp("4.4a")

% load data
load prostateStnd.mat

% calc train data norm vectors and apply to test data
% obtain mean and covariance of training data
%flip data to conform with dxn matrix size
disp("Transpose data to put it in Dxn format")
Xtrain = Xtrain';
Xtest = Xtest';
ytrain = ytrain';
ytest = ytest';

meanX = XYmean(Xtrain);
meanY = mean(ytrain);
SX = Xcov(Xtrain);
SY = Xcov(ytrain);
disp("muX")
disp(meanX)
disp("muy")
disp(meanY)
disp("SX")
disp(SX)
disp("SY")
disp(SY)
%mean_vec = 
%std_vec  =

% Normalize data:
ytrain_normalized = ytrain - meanY;
ytest_normalized  = ytest - meanY;

Xtrain_normalized = Xtrain - meanX;
Xtest_normalized = Xtest - meanX;

meanX_normalized = XYmean(Xtrain_normalized);
meanY_normalized = mean(ytrain_normalized);
SX = Xcov(Xtrain_normalized);
SY = Xcov(ytrain_normalized);

SXY = XYcov(Xtrain_normalized, ytrain_normalized);
SYX = XYcov(ytrain_normalized, Xtrain_normalized);

%% 4.4 b) Train ridge regression model for various lambda (regularization parameter) values

disp("4.4b")

xx = [-5:10];
lambda_vec = exp(xx);
coefficient_mat = zeros(size(Xtrain_normalized,1),length(lambda_vec));
% coefficient mat = Dxn matrix
% d = features
% n = lambda_vec length
% columns are each a w_ridge for the given lamdba_vec value

disp('Iterating through different lambdas...')
for i = 1:length(lambda_vec)
    coefficient_mat(:,i) = inv(diag(ones(size(coefficient_mat,1),1)*lambda_vec(i)) + SX) * SXY;
end
coefficient_mat

% b_ridge given the w_ridge column of coefficient_mat
% = 0 approx. since data is normalized
b_ridge = meanY_normalized  - (coefficient_mat' * meanX_normalized)
disp('4.4b Done.')
%% 4.4 c) Plotting ridge regression coefficients

disp("4.4c")

figure, grid on; hold on; xlabel('ln(lambda)'), ylabel('Feature coefficient values'); title('Feature coefficent values (w) for various regularization amounts')
mycolor = ["#0072BD" "#D95319" "#EDB120" "#7E2F8E" "#77AC30" "#4DBEEE" "#A2142F" "#00FF00"];
for i = 1:size(coefficient_mat,1)
    p = plot(coefficient_mat(i,:));
    p.Color = mycolor(i);
end
legend(names(1:8));
hold off
%Discuss what happens to the coefficients as λ becomes larger
    % as lambda increases, the value of lambda ||w||^2 
    % this means that larger values of w are penalized more heavily.
    % this promotes using a smaller value for w which is less complete (data wise). 
    % a precise answer using a complete w exists, but cannot be used
    % because of the penalty imposed.
    % the best CCR for the training will overfitting the data
    % large lambda penalty mitigates overfitting the data
    % small lambda penalty mitigates underfitting the data
    % i.e. a the smallest lambda will overfit the data, and the largest lambda
    % will underfit the data, leading to unoptimized test CCR on both sides
    % this is akin to the lambda constraint we used when implementing DP-centers
%% 4.4 d) Plotting MSE values as function of ln(lambda)

disp("4.4d")

%calculate w_ridge using lambda coefficient matrix
w_ridge = coefficient_mat

%calculate b_ridge using w_ridge
b_ridge = meanY_normalized  - (w_ridge' * meanX_normalized )

MSE_train = zeros(1,size(coefficient_mat,2));
MSE_test = zeros(1,size(coefficient_mat,2));
for i = 1:size(MSE_train,2)
    MSE_train(i) = MSE(Xtrain_normalized,ytrain_normalized,w_ridge(:,i),b_ridge(i,1));
    MSE_test(i) = MSE(Xtest_normalized,ytest_normalized,w_ridge(:,i),b_ridge(i,1));
end
MSE_train
MSE_test
figure, grid on; hold on; xlabel('ln(lambda)'), ylabel('MSE'); title('MSE train')
plot(MSE_train)
figure, grid on; hold on; xlabel('ln(lambda)'), ylabel('MSE'); title('MSE test')
plot(MSE_test)
figure, grid on; hold on; xlabel('ln(lambda)'), ylabel('MSE');
plot(MSE_train)
plot(MSE_test)
legend({"train_norm","test_norm"},  'Location','northwest')

% Discuss your observations.
    % As stated in 4.4c, 
    % large lambda penalty mitigates overfitting the data
    % small lambda penalty mitigates underfitting the data
    % Overfitting the data (small lambda) will produce the best training
    % set results, but not the best test results
    % likewise, underfitting the data will produce poor training results,
    % and likely poor test results
    % the best choice will be somewhere in the middle, to prevent
    % overfitting and underfitting from occuring. 
    % Here we see ln(lambda_vec(5)) produces the lowest MSE on the test
    % data, and is the preferred lambda value to use. 
%%
function MSE = MSE(data, label, w, b)
%takes in data, w, b and calcualtes the MSE
    MSE_array = zeros(1,size(data,2));
    for i = 1:size(data,2)
        MSE_array(i) = (label(i) - w'*data(:,i) - b)^2;
    end
    MSE = 1/(size(data,2))*sum(MSE_array);
end
%% Helper functions from 4.3

function mean = XYmean(data)
% takes in an array of data (x1; x2; x3; ...)
% returns the mean of each column
    mean = zeros(size(data,1),1);
    for i = 1:size(data,1)
        Imean = sum(data(i,:))/size(data,2);
        mean(i,1) = Imean;
    end
end

function data_cov = Xcov(data)
% takes in an array of data (x1; x2; x3; ...)
% returns covariance of the data, dxd matrix
    mean = XYmean(data);
    data_norm = (data(:, :) - mean);
    data_cov = 0;
    for i = 1:size(data_norm,2)
        data_cov = data_cov + data_norm(:,i)*transpose(data_norm(:,i));
    end
    data_cov = data_cov / size(data_norm,2);
end

function data_cov = XYcov(data1,data2)
% takes in 2 arrays of data
% returns covariance of the data
    mean1 = XYmean(data1);
    mean2 = XYmean(data2);
    data1_norm = (data1(:, :) - mean1);
    data2_norm = (data2(:, :) - mean2);
    data_cov = 0;
    for i = 1:size(data1_norm,2)
        data_cov = data_cov + data1_norm(:,i)*transpose(data2_norm(:,i));
    end
    data_cov = data_cov / size(data1_norm,2);
end