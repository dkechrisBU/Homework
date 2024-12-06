%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ENG EC 503 (Ishwar) Fall 2023
% HW 8.2-8.3
% <Your full name and BU email>
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Kernel K-Means

clear, clc, close all,

%% Load Data
% choose which dataset to load (.mat files)
% dataset = "2-class-concentric"; % or "3-class-moon-data"
% K = 2;                          % number of clusters
% sigma = .16;                    % choice of bandwidth for rbf

dataset = "3-class-moon-data";
K = 3;                          % number of clusters
sigma = 8;                     % choice of bandwidth for rbf

N = 500;                        % number of points per cluster

load(dataset)
plot(data(:,1), data(:,2), '*')
title("Dataset")

%% Vanilla K-means

% TODO: create random initialization for mu

[MU_final, WCSS, y_hat2] = k_means(Mu, data, N, K,2);

%% Kernel K-means

% TODO: create random initialization for alpha

%[Alpha_final, WCSS, y_hat] = kernel_k_means(Alpha, data, N, K, sigma);



