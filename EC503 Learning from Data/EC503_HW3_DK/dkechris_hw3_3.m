f% EC 503 - HW 3 - Fall 2023
% K-Means starter code

clear, clc, close all;
rng('default');
defaultseed = rng;

%last part of question is dependent on the midpoint, will fall into one of
%the two buckets, compute the likelihood of it
%hard way is compute integral of pdf
%easy way is to do 2d picture

%% Generate Gaussian data:
% Add code below:

%gen std normal g then shift data to new mean


%create 3 distributions of μ1 = [2, 2]T, μ2 = [−2, 2]T,μ3 = [0, −3.25]T, and Σ1 = 0.02 · I2, Σ2 = 0.05 · I2, Σ3 = 0.07 · I2, where I2 is the 2 × 2 identity matrix
m1 = [2, 2]
m2 = [-2, 2]
m3 = [0,-3.25]
identity = [1 0; 0 1]
s1 = 0.02*identity
s2 = 0.05*identity
s3 = 0.07*identity
g1 = random(gmdistribution(m1,s1),50)
g2 = random(gmdistribution(m2,s2),50)
g3 = random(gmdistribution(m3,s3),50)
scatter(g1(:,1),g1(:,2),"red")
hold on
scatter(g2(:,1),g2(:,2),"green")
scatter(g3(:,1),g3(:,2),"blue")
hold off

gall = [g1;g2;g3]

%% 3.3 

% 3.3)
% create 100 random points between -4 and 4
rx = -4 + (4+4)*rand(100,1)
ry = -4 + (4+4)*rand(100,1)
rxy = [rx ry]

k_WCSS = zeros(10)
k_WCSS_best = zeros(1,10)
for k = 2:10
    k
    k_WCSS_best(1,k) = 10000
    for i = 1:k:k*10
        mu_init = rxy(i:i+k-1,:);
        j = 1+(i-1)/k;
        output = kmeansimp(k,mu_init,gall);
        k_WCSS(j,k) = output;
        if output < k_WCSS_best(1,k)
            k_WCSS_best(1,k) = output;
            %k_mu_best(:,k) = mu_init
            k_j_best(1,k) = j;
        end
    end
end

k_WCSS
k_WCSS_best
%k_WCSS_best = [0  418.0056   13.2883   11.2170    9.7814    9.4245
% 7.9506    6.2270    7.6788    8.2860]
k_WCSS_best = min(k_WCSS)
%k_mu_best
k_j_best
lambda = [15 20 25 30]

data = zeros(size(k_WCSS_best,2),size(lambda,2))
for i = 2:size(k_WCSS_best,2)
    for j = 1:size(lambda,2)
        data(i, j) = k_WCSS_best(i) + i*lambda(j)
    end
end

plot(data)

%plotkvslambda(data)
%% 3.4 

%dp means implementation

%% K-Means implementation
% Add code below

function WCSS = kmeansimp(k,MU_init,data)
    MU_previous = MU_init;
    MU_current = MU_init;
    
    % initializations
    labels = ones(size(data,1),1);
    converged = 0;
    iteration = 0;
    convergence_threshold = 0.025;

    while (converged==0)
        MU_previous = MU_current;
        iteration = iteration + 1;
        fprintf('Iteration: %d\n',iteration)

        %calculate which clusters all points belong to
        centerl = addlabeltocenter(MU_current);
        labels = knn(data,centerl,1);
        datalabels = [data labels];
        %calculate best centers (centroids) for each cluster
        %average of all points with that label

        %set new centers to MU_current
        new_centers = getnewcenters(datalabels, k);
        MU_current = new_centers

        %CODE 4 - Check for convergence
        %if largest of (dist new - dist old) for each
        largestdelta = large_diff(MU_previous,MU_current);
        if (largestdelta < convergence_threshold)
            converged=1;
        end
    
        %CODE 5 - Plot clustering results if converged:
        if (converged == 1)
            fprintf('\nConverged.\n')
            
            %If converged, get WCSS metric
            %sum of all distances^2 for each point in cluster
            WCSS = 0
          
            for i = 1:k
                Xyi = datalabels(datalabels(:,3)==i,:);
                WCSS = WCSS + WCSoS(MU_current(i,:),Xyi);
            end
            
            if k == 3
                Xy1 = datalabels(datalabels(:,3)==1,:);
                Xy2 = datalabels(datalabels(:,3)==2,:);
                Xy3 = datalabels(datalabels(:,3)==3,:);
                scatter(Xy1(:,1),Xy1(:,2),"red")
                hold on
                scatter(Xy2(:,1),Xy2(:,2),"green")
                scatter(Xy3(:,1),Xy3(:,2),"blue")
                scatter(MU_current(:,1),MU_current(:,2),"black")
                hold off
            else
                datalabels = sortrows(datalabels,3)
                scatter(datalabels(:,1),datalabels(:,2),[],datalabels(:,3))
                hold on
                scatter(MU_current(:,1),MU_current(:,2),"black")
                colormap("jet")
                hold off
            end


        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
end

%% sample_circle

function [data ,label] = sample_circle( num_cluster, points_per_cluster )
% Function to sample 2-D circle-shaped clusters
% Input:
% num_cluster: the number of clusters 
% points_per_cluster: a vector of [num_cluster] numbers, each specify the
% number of points in each cluster 
% Output:
% data: sampled data points. Each row is a data point;
% label: ground truth label for each data points.
%
% EC 503: Learning from Data
% Fall 2022
% Instructor: Prakash Ishwar
% HW 3, Problem 3.2(f) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if nargin == 0
      num_cluster = 2;
      points_per_cluster = 500*ones(num_cluster,1);
    end
    if nargin == 1
       points_per_cluster = 500*ones(num_cluster,1);
    end
    points_per_cluster=points_per_cluster(:);
    
    data = zeros([sum(points_per_cluster), 2]);
    label = zeros(sum(points_per_cluster),1);
    idx = 1;
    bandwidth = 0.1;
    
    for k = 1 : num_cluster
        theta = 2 * pi * rand(points_per_cluster(k), 1);
        rho = k + randn(points_per_cluster(k), 1) * bandwidth;
        [x, y] = pol2cart(theta, rho);
        data(idx:idx+points_per_cluster(k)-1,:) = [x, y];
        label(idx:idx+points_per_cluster(k)-1)=k;
        idx = idx + points_per_cluster(k);
    end
end
%% Helper functions:

function largestdiff = large_diff(a1, a2)
% take an array of points, a1 and a2, where both are the same dim X:2
% outputs the largest difference between the distance of each set in a1-a2 array 
% if not same size, outputs 1
    if size(a1) == size(a2)
        adist = zeros(size(a1,1));
        %calculate distance for each row of points
        for i = 1:size(a1,1)
            adist(i) = dist(a1(i,:),a2(i,:));
        end
        %take largest of distances
        largestdiff = max(adist);
    else
        largestdiff = 1
    end
end

function dist = dist(p1, p2)
% takes in 2 points and outputs the distance between them
% input p1, p2: 1:2 array with x,y coordinates
% outputs distance between them 
    dist = sqrt( power(p1(1) - p2(1),2) + power(p1(2) - p2(2),2));
end

function arr = addlabeltocenter(centers)
% append labels to centers
% input: array of centers
% output: array of centers, appending the row number to each row
    labels = zeros(size(centers,1),1);
    for i = 1:size(centers,1)
        labels(i) = i;
    end
    arr = [centers labels];
end

function centers = getnewcenters(data, k)
% this function takes in an array of data with labels attached to it and
% calculates the centers associated with them
% input data: an x:3 array of x,y,label 
% output centers: array of l centers where l is the max label
    centers = zeros(k,2);
    for i = 1:k
        Xyi = data(data(:,3)==i,:);
        x_val = sum(Xyi(:,1)/size(Xyi,1));
        y_val = sum(Xyi(:,2)/size(Xyi,1));
        centers(i,:) = [x_val, y_val];
    end
end

function SoS = WCSoS(center, data)
%returns the sum of square distance to the center
    SoS = 0;
    fulldist = calcdist(center, data);
    dist = fulldist(:,4);
    dist2 = dist.*dist;
    SoS = sum(dist2,1);
end

%can I just use Knn where k = 1 and points is the centers?????
%looks like yes
function label = assignPoints(points, centers)
% takes in an array of points, an array of (centers and labels)
% outputs an array size(points) giving corresponding label for each of point
    label = zeros(size(points,1),1);
    for i = 1:size(points,1)
        dist = calcdist(points(i,:), centers);
        dist = sortrows(dist,3);
        nearest = dist(1,:);
        label(i) = nearest(3);
    end
end

function plotkvslambda(data)
%takes an array of lambda rows and k columns to plot

end

%HW2, check to see if all functions needed
function label = knn(points, arr, k)
% takes in an array of points, an array of data+labels, and k
% outputs an array size(points) giving corresponding label for each of point
    label = zeros(size(points,1),1);
    for i = 1:size(points,1)
        dist = calcdist(points(i,:), arr);
        dist = sortrows(dist,4);
        nearest = dist(1:k,:);
        if k == 1
            label(i) = nearest(3);
        else
            l = mode(nearest);
            l = l(3);
            label(i) = l;
        end
    end
end

function prob = knnProb(points, arr, k, label)
% takes in an array of points, an array of data+labels, k, label for prob
% outputs prob each of points belongs to label
    prob = zeros(size(points,1),1);
    for i = 1:size(points,1)
        dist = calcdist(points(i,:), arr);
        dist = sortrows(dist,4);
        nearest = dist(1:k,:);
        n = nearest(:,3);
        count = 0;
        for j = 1:size(n,1)
            if n(j) == label
                count = count+1;
            end
        end
        p = count / k;
        prob(i) = p;
    end
end

function dist = calcdist(point, arr)
% takes point, an array of data + labels (labels not required)
% outputs an array of distance from arr to the point 
% as the next column in arr
    d = sqrt( power(arr(:,1) - point(1),2) + power(arr(:,2) - point(2),2));
    dist = [arr d];
end

function a = LOOCV(tset, k)
% takes a test set and a k value
% computes LOOCV for that test set using kNN=k
% returns a = misclassification rate for k
tsize = size(tset,1);
a = 0;
    for i = 1:tsize
        testX = tset(i,:);
        if i == 1
            tempset = [(tset(i+1:tsize,:))];
        elseif i == tsize
            tempset = [tset(1:i-1,:)];
        else
            tempset = [tset(1:i-1,:); (tset(i+1:tsize,:))];
        end
        testLabel = knn(testX(1:2),tempset,k);
        %if test label doesn't match real label
        if testLabel ~= testX(3)
            a = a+1;
        end
    end
a = a/tsize;
end