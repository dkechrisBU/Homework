%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ENG EC 503 (Ishwar) Fall 2023
% HW 7.1
% <Your full name and BU email>
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear, clc, close all,
%% load iris dataset (3 classes, 50 observations each) and plot pairs of features:

load iris.mat
%% Plot train and test data:

rng('default')

%{
%assign +/-
Y_label2 = zeros(size(Y_label_train,1),1)
for i_class = 1:size(Y_label2,1)
    if Y_label_train(i_class,1) == 1
        Y_label2(i_class,1) = 1
    else
        Y_label2(i_class,1) = 0
    end
end
Y_label_train = [Y_label_train Y_label2]

Y_label2 = zeros(size(Y_label_test,1),1)
for i_class = 1:size(Y_label2,1)
    if Y_label_test(i_class,1) == 1
        Y_label2(i_class,1) = 1
    else
        Y_label2(i_class,1) = 0
    end
end
Y_label_test = [Y_label_test Y_label2]

X_data_train = [X_data_train(:,2) X_data_train(:,4)]
X_data_test = [X_data_test(:,2) X_data_test(:,4)]
%}
X_data_train = [X_data_train(:,2) X_data_train(:,4)]
X_data_test = [X_data_test(:,2) X_data_test(:,4)]
%% train pairwise models:

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [1,2] pair -- CLASS 1 = -1, CLASS 2 = +1
%add ext notation
Xy_test = [X_data_test ones(size(X_data_test,1),1) Y_label_test]
Xy_test_1 = Xy_test(Xy_test(:,4)==1,:);
Xy_test_2 = Xy_test(Xy_test(:,4)==2,:);
Xy_test_3 = Xy_test(Xy_test(:,4)==3,:);

Xy_train = [X_data_train ones(size(X_data_train,1),1) Y_label_train]
Xy_train_1 = Xy_train(Xy_train(:,4)==1,:);
Xy_train_2 = Xy_train(Xy_train(:,4)==2,:);
Xy_train_3 = Xy_train(Xy_train(:,4)==3,:);

figure 
scatter(Xy_train_1(:,1),Xy_train_1(:,2),"filled","r")
hold on 
scatter(Xy_train_2(:,1),Xy_train_2(:,2),"filled","g")
scatter(Xy_train_3(:,1),Xy_train_3(:,2),"filled","b")
hold off

figure
scatter(Xy_test_1(:,1),Xy_test_1(:,2),"filled","r")
hold on 
scatter(Xy_test_2(:,1),Xy_test_2(:,2),"filled","g")
scatter(Xy_test_3(:,1),Xy_test_3(:,2),"filled","b")
hold off



%for pair 1,2
XY_test_12 = [Xy_test_1; Xy_test_2]
XY_train_12 = [Xy_train_1; Xy_train_2]

%assign +/-
% class 1 = -1
% class 2 = 1
Y_label2 = zeros(size(XY_train_12,1),1);
for i_class = 1:size(Y_label2,1)
    if XY_train_12(i_class,4) == 1
        Y_label2(i_class,1) = -1;
    else
        Y_label2(i_class,1) = 1;
    end
end
XY_train_12 = [XY_train_12 Y_label2]

Y_label2 = zeros(size(XY_test_12,1),1);
for i_class = 1:size(Y_label2,1)
    if XY_test_12(i_class,4) == 1
        Y_label2(i_class,1) = -1;
    else
        Y_label2(i_class,1) = 1;
    end
end
XY_test_12 = [XY_test_12 Y_label2]

% stochastic subgradient descent
C = 1.2;
t_max = 2 * power(10,5);
n = size(XY_train_12,1);
datetime()
[theta_12, sample_norm_cost_12, training_CCR_12, test_CCR_12] = SSGD(C, t_max, XY_train_12, XY_test_12)

datetime()

%%%%%%%%%%%%%%%%
% Visualize separating hyperplane between the two classes:
%hplaneplot(theta)
%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PLOTS and SUMMARIZING VALUES FOR CLASSES 1,2 PAIR %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Plot cost
figure
plot(sample_norm_cost_12)
xlabel("iteration # = t*1000")
title("The normalized loss for training set 12")
ylabel("Normalized loss")

% Plot train ccr
figure
plot(training_CCR_12)
xlabel("iteration # = t*1000")
title("CCR for training set 12")
ylabel("Training CCR")

% Plot test ccr
figure
plot(test_CCR_12)
xlabel("iteration # = t*1000 12")
title("CCR for test set 12")
ylabel("Test CCR")

% final values
theta_12
sample_norm_cost_12(200,1)
training_CCR_12(200,1)
test_CCR_12(200,1)

disp('(d)(iv): Theta_12 Training confusion matrix:')
disp('rows are the predicted label, columns are actual label')
conf_mat_12 = conf_mat_bin(XY_train_12, theta_12)
disp('(d)(v): Theta_12 Test confusion matrix:')
disp('rows are the predicted label, columns are actual label')
conf_mat_test_12 = conf_mat_bin(XY_test_12, theta_12)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [1,3] pair -- CLASS 1 = -1, CLASS 3 = +1
XY_test_13 = [Xy_test_1; Xy_test_3]
XY_train_13 = [Xy_train_1; Xy_train_3]

%assign +/-
% class 1 = -1
% class 3 = 1
Y_label2 = zeros(size(XY_train_13,1),1);
for i_class = 1:size(Y_label2,1)
    if XY_train_13(i_class,4) == 1
        Y_label2(i_class,1) = -1;
    else
        Y_label2(i_class,1) = 1;
    end
end
XY_train_13 = [XY_train_13 Y_label2]

Y_label2 = zeros(size(XY_test_13,1),1);
for i_class = 1:size(Y_label2,1)
    if XY_test_13(i_class,4) == 1
        Y_label2(i_class,1) = -1;
    else
        Y_label2(i_class,1) = 1;
    end
end
XY_test_13 = [XY_test_13 Y_label2]

% stochastic subgradient descent
C = 1.2;
t_max = 2 * power(10,5);
n = size(XY_train_13,1);
datetime()
[theta_13, sample_norm_cost_13, training_CCR_13, test_CCR_13] = SSGD(C, t_max, XY_train_13, XY_test_13)

datetime()


%%%%%%%%%%%%%%%%
% Visualize separating hyperplane between the two classes:


%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PLOTS and SUMMARIZING VALUES FOR CLASSES 1,3 PAIR %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Plot cost
figure
plot(sample_norm_cost_13)
xlabel("iteration # = t*1000")
title("The normalized loss for training set 13")
ylabel("Normalized loss")

% Plot train ccr
figure
plot(training_CCR_13)
xlabel("iteration # = t*1000")
title("CCR for training set 13")
ylabel("Training CCR")

% Plot test ccr
figure
plot(test_CCR_13)
xlabel("iteration # = t*1000")
title("CCR for test set 13")
ylabel("Test CCR")

% final values
theta_13
sample_norm_cost_13(200,1)
training_CCR_13(200,1)
test_CCR_13(200,1)

disp('(d)(iii): Theta_13 Training confusion matrix:')
disp('rows are the predicted label, columns are actual label')
conf_mat_13 = conf_mat_bin(XY_train_13, theta_13)
disp('(d)(iv): Theta_13 Test confusion matrix:')
disp('rows are the predicted label, columns are actual label')
conf_mat_test_13 = conf_mat_bin(XY_test_13, theta_13)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [2,3] pair -- CLASS 2 = -1, CLASS 3 = +1
XY_test_23 = [Xy_test_2; Xy_test_3]
XY_train_23 = [Xy_train_2; Xy_train_3]

%assign +/-
% class 2 = -1
% class 3 = 1
Y_label2 = zeros(size(XY_train_23,1),1);
for i_class = 1:size(Y_label2,1)
    if XY_train_23(i_class,4) == 2
        Y_label2(i_class,1) = -1;
    else
        Y_label2(i_class,1) = 1;
    end
end
XY_train_23 = [XY_train_23 Y_label2]

Y_label2 = zeros(size(XY_test_23,1),1);
for i_class = 1:size(Y_label2,1)
    if XY_test_23(i_class,4) == 2
        Y_label2(i_class,1) = -1;
    else
        Y_label2(i_class,1) = 1;
    end
end
XY_test_23 = [XY_test_23 Y_label2]

% stochastic subgradient descent
C = 1.2;
t_max = 2 * power(10,5);
n = size(XY_train_23,1);
datetime()
[theta_23, sample_norm_cost_23, training_CCR_23, test_CCR_23] = SSGD(C, t_max, XY_train_23, XY_test_23)

datetime()


%%%%%%%%%%%%%%%%
% Visualize separating hyperplane between the two classes:


%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PLOTS and SUMMARIZING VALUES FOR CLASSES 2,3 PAIR %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Plot cost
figure
plot(sample_norm_cost_23)
xlabel("iteration # = t*1000")
title("The normalized loss for training set 23")
ylabel("Normalized loss")

% Plot train ccr
figure
plot(training_CCR_23)
xlabel("iteration # = t*1000")
title("CCR for training set 23")
ylabel("Training CCR")

% Plot test ccr
figure
plot(test_CCR_23)
xlabel("iteration # = t*1000")
title("CCR for test set 23")
ylabel("Test CCR")

% final values
theta_23
sample_norm_cost_23(200,1)
training_CCR_23(200,1)
test_CCR_23(200,1)

disp('(d)(iii): Theta_23 Training confusion matrix:')
disp('rows are the predicted label, columns are actual label')
conf_mat_23 = conf_mat_bin(XY_train_23, theta_23)
disp('(d)(iv): Theta_23 Test confusion matrix:')
disp('rows are the predicted label, columns are actual label')
conf_mat_test_23 = conf_mat_bin(XY_test_23, theta_23)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% train pairwise models: 3 class classifier using OVO method (ALL PAIRS) (use the thetas from the previous methods)

% ALL thetas --> theta_12, theta_13, theta_23
% Data is in:  X_train, Y_train, X_test, Y_test


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% e i) Training data
datetime()

y_predict_12 = sign(theta_12'*Xy_train(:,1:3)')';
y_predict_13 = sign(theta_13'*Xy_train(:,1:3)')';
y_predict_23 = sign(theta_23'*Xy_train(:,1:3)')';
y_predict = zeros(size(Xy_train,1),3);

CCR = 0;
confusion_matrix = zeros(3);
for i = 1:size(y_predict_12,1)
    %check theta_12
    if y_predict_12(i) > 0
        y_predict(i,2) = y_predict(i,2) +1;
    else
        y_predict(i,1) = y_predict(i,1) +1;
    end
    %check theta_13
    if y_predict_13(i) > 0
        y_predict(i,3) = y_predict(i,3) +1;
    else
        y_predict(i,1) = y_predict(i,1) +1;
    end
    %check theta_23
    if y_predict_23(i) > 0
        y_predict(i,3) = y_predict(i,3) +1;
    else
        y_predict(i,2) = y_predict(i,2) +1;
    end
    
    [~,yguess] = max(y_predict(i,:));
    if yguess == Xy_train(i,4)
        CCR = CCR +1;
    end

    confusion_matrix(yguess, Xy_train(i,4)) = confusion_matrix(yguess, Xy_train(i,4)) + 1;

end 
CCR = CCR / size(Xy_train,1)
confusion_matrix
datetime()

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% e ii) Testing data
y_predict_test_12 = sign(theta_12'*Xy_test(:,1:3)')';
y_predict_test_13 = sign(theta_13'*Xy_test(:,1:3)')';
y_predict_test_23 = sign(theta_23'*Xy_test(:,1:3)')';
y_predict_test = zeros(size(Xy_test,1),3);

CCR_test = 0;
confusion_matrix_test = zeros(3);
for i = 1:size(y_predict_test_12,1)
    %check theta_12
    if y_predict_test_12(i) > 0
        y_predict_test(i,2) = y_predict_test(i,2) +1;
    else
        y_predict_test(i,1) = y_predict_test(i,1) +1;
    end
    %check theta_13
    if y_predict_test_13(i) > 0
        y_predict_test(i,3) = y_predict_test(i,3) +1;
    else
        y_predict_test(i,1) = y_predict_test(i,1) +1;
    end
    %check theta_23
    if y_predict_test_23(i) > 0
        y_predict_test(i,3) = y_predict_test(i,3) +1;
    else
        y_predict_test(i,2) = y_predict_test(i,2) +1;
    end

    [~,yguess] = max(y_predict_test(i,:));
    if yguess == Xy_test(i,4)
        CCR_test = CCR_test +1;
    end
    confusion_matrix_test(yguess, Xy_test(i,4))  = confusion_matrix_test(yguess, Xy_test(i,4)) + 1;
end
CCR_test = CCR_test / size(Xy_test,1)
confusion_matrix_test
datetime()

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SUMMARIZING VALUES 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Final plot for decision boundary in the 3 class case:

%for l1 = 1:length(x_range)
%    for l2 = 1:length(y_range)
        
        % 1,2 classifier
        
        % 1,3 classifier
        
        % 2,3 classifier
        
        % Condition check -- if all different, randomly choose one; if not,
        % choose mode:

%    end
%end

%%
function [theta, sample_norm_cost, training_CCR, test_CCR]  = SSGD(C, t_max, XY_train, XY_test)
    theta = [0; 0; 0];
    n = size(XY_train,1);
    Xext = XY_train(:,1:3)
    Ylabel = XY_train(:,5)
    Xtestext = XY_test(:,1:3)
    Ytestlabel = XY_test(:,5)
    datetime()
    sample_norm_cost = zeros(t_max/1000,1)
    training_CCR = zeros(t_max/1000,1);
    test_CCR = zeros(t_max/1000,1);

    for iii = 1:t_max
        % current iteration index
        t = iii; 
        st = .5/t;
        % choose sample index:   
        j = randi(n);
        xjext = XY_train(j,1:3);
        yj = XY_train(j,5);
        
        % compute subgradient:
        % v with Id for d=2   
        v = [theta(1:2,1); 0];
        if yj * theta' * xjext' < 1
            v = v - n * C * yj * xjext';
        end
        % update parameters:
        theta = theta - st*v;
        
        % redefine parameters
        
        if (mod(t,1000) == 0)
            Ylabel;
            theta';
            Xext';
            %Xext'*theta'
            %Xext' * theta' * Ylabel
            Ylabel .* (theta' * Xext')';
            sum( C * Ylabel .* (theta' * Xext')');
            %C * max(0,1 - Ylabel * theta' * Xext')
            %sum(C * max(0,1 - Ylabel * theta' * Xext'))
            %sum(sum(transpose(C * max(0,1 - ylabel * theta' * text'))))
            %sample_norm_cost(t/1000,1) = norm(theta(1:2))/2 + sum(transpose(C * max(0,1 - Ylabel * theta' * Xext')));
            sample_norm_cost(t/1000,1) = norm(theta(1:2))/2 + sum( C * Ylabel .* (theta' * Xext')');
            
            % Training CCR:
            training_CCR(t/1000,1) = tCCR(theta, Xext, Ylabel);
            % Test CCR:
            test_CCR(t/1000,1) = tCCR(theta, Xtestext, Ytestlabel);

            
        end
    end
   datetime() 
end

function CCR = tCCR(theta, Xext, Ylabel)
    CCR = 0;
    n = size(Xext,1);
    y_predict = sign(theta'*Xext')';
    CCR = sum(y_predict == Ylabel) / n;
    %for i = 1:n
    %    if Ylabel(i,1) == sign(theta'*Xext(i,:)')
    %        CCR = CCR + 1
    %    end
    %end
    %CCR = CCR / n
end

function conf_mat = conf_mat_bin(XY, theta)
% function takes 2 data sets, w and b
% returns the correct classification rate of the data using +/-w and b
% returns a confusion matrix where the rows are the predicted rows and the
% columns are the correct columns
    n = size(XY,1);
    x_ext = XY(:,1:3);
    label = XY(:,5);
    conf_mat = zeros(3);
    
    % calc conf map
    for j = 1:n
        xj = x_ext(j,:)';
        yj = sign(theta' * xj);
        yactual = label(j,1);
        conf_mat(yj+2,yactual+2) = conf_mat(yj+2,yactual+2) + 1;
    end
    conf_mat(2,:) = [];
    conf_mat(:,2) = [];
end