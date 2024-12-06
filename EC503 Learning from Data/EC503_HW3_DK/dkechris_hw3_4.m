% EC 503 - HW 3 - Fall 2023
% DP-Means starter code

clear, clc, close all;
rng('default');
defaultseed = rng;
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
%% Generate NBA data:
% Add code below:

% HINT: readmatrix might be useful here
nba = readmatrix("NBA_stats_2018_2019.xlsx");
mpg = nba(:,5)
ppg = nba(:,7)
mppg = [mpg ppg]

%scatter(mpg,ppg)
%% DP Means method:

% Parameter Initializations
%LAMBDA = 0.15;
%convergence_threshold = 1;
%num_points = length(DATA);
%total_indices = [1:num_points];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 3.4a)
% the role of lambda is to add a penalty for having too many clusters
% the penalty is proportional to the number of clusters (k * lambda)
% prevents ideal k from being all data points

% 3.4b) 
% lambda = [.15; .4; 3; 20]
% outputk = zeros(4,1)
% for i = 1:size(lambda,1)
%     outputk(i) = dpmeansimp(lambda(i),gall)
% end

%so each output has a graph
% output1 = dpmeansimp(.15,gall)
% output2 = dpmeansimp(.4,gall)
% output3 = dpmeansimp(3,gall)
% output4 = dpmeansimp(20,gall)


% 3.4c) 
output1 = dpmeansimp(44,mppg)
output2 = dpmeansimp(100,mppg)
output3 = dpmeansimp(450,mppg)

%% DP Means - Initializations for algorithm %%%
% cluster count

function k = dpmeansimp(lambda, data)
    k = 1;
    current = 1;
    labels = ones(size(data,1),1);
    freshdata = [data labels];
    MU_current = getnewcenters(freshdata,1);
    MU_previous = MU_current;
    num_points = length(data);
    
    % initializations
    converged = 0;
    iteration = 0;
    convergence_threshold = 1;
    %skeleton
    %{
    K = 1;
    
    % sets of points that make up clusters
    L = {};
    L = [L [1:num_points]];
    
    % Class indicators/labels
    Z = ones(1,num_points);
    
    % means
    MU = [];
    MU = [MU; mean(DATA,1)];
    %MU = [MU; mean(DATA,1) mean(DATA,2)];
    %}
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    while (converged == 0)
        MU_previous = MU_current;
        iteration = iteration + 1;
        fprintf('Current iteration: %d...\n',iteration)
        % Per Data Point:
        for i = 1:num_points
            
            % CODE 1 - Calculate distance from current point to all % currently existing clusters
            distfull = calcdist(data(i,:),MU_current);
            dist = distfull(:,size(distfull,2));
    
            % CODE 2 - Look at how the min distance of the cluster distance list compares to LAMBDA
            if min(dist)*min(dist) > lambda
               MU_current = [MU_current; data(i,:)];
               current = current + 1;
            end
            labels(i) = current;
        end
    
        % calculate which clusters all points belong to
        %CODE 3 - Form new sets of points (clusters)
        centerl = addlabeltocenter(MU_current);
        labels = knn(data,centerl,1);
        datalabels = [data labels];
    
    
        %CODE 4 - Recompute means per cluster
        % calculate best centers (centroids) for each cluster
        % average of all points with that label
        new_centers = getnewcenters(datalabels, current);
    
        %remove empty clusters
        for i = 1:size(new_centers,1)
            Xyi = datalabels(datalabels(:,3)==i,:);
            if size(Xyi,1) == 0
                new_centers(i,:) = [];
                i = i - 1;
                current = current - 1;
            end
        end
        
        %set new centers to MU_current
        MU_current = new_centers;
        k = current;
        
        %CODE 5 - Test for convergence: number of clusters doesn't change and means stay the same %%%
        %if largest of (dist new - dist old) for each
        largestdelta = large_diff(MU_previous,MU_current);
        if (largestdelta < convergence_threshold)
            converged=1;
        end    
    
        %CODE 6 - Plot final clusters after convergence
        %Write code below here:
        
        if (converged == 1)
            fprintf('\nConverged.\n')
            
            %If converged, get WCSS metric
            %sum of all distances^2 for each point in cluster
            WCSS = 0
          
            for i = 1:k
                Xyi = datalabels(datalabels(:,3)==i,:);
                WCSS = WCSS + WCSoS(MU_current(i,:),Xyi);
            end
            WCSS = WCSS + k*lambda
            
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
                datalabels = sortrows(datalabels,3);
                scatter(datalabels(:,1),datalabels(:,2),[],datalabels(:,3))
                hold on
                scatter(MU_current(:,1),MU_current(:,2),"black")
                colormap("jet")
                hold off
            end
        end   
    end
end



%helper functions
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

function dist = calcdist(point, arr)
% takes point, an array of data + labels (labels not required)
% outputs an array of distance from arr to the point 
% as the next column in arr
    d = sqrt( power(arr(:,1) - point(1),2) + power(arr(:,2) - point(2),2));
    dist = [arr d];
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

function largestdiff = large_diff(a1, a2)
% take an array of points, a1 and a2, where both are the same dim X:2
% outputs the largest difference between the distance of each set in a1-a2 array 
% if not same size, outputs 1
    if size(a1) == size(a2)
        adist = zeros(size(a1,1),1);
        %calculate distance for each row of points
        for i = 1:size(a1,1)
            adist(i) = dist(a1(i,:),a2(i,:));
        end
        %take largest of distances
        largestdiff = max(adist);
    else
        largestdiff = 10;
    end
end

function dist = dist(p1, p2)
% takes in 2 points and outputs the distance between them
% input p1, p2: 1:2 array with x,y coordinates
% outputs distance between them 
    dist = sqrt( power(p1(1) - p2(1),2) + power(p1(2) - p2(2),2));
end

function SoS = WCSoS(center, data)
%returns the sum of square distance to the center
    SoS = 0;
    fulldist = calcdist(center, data);
    dist = fulldist(:,4);
    dist2 = dist.*dist;
    SoS = sum(dist2,1);
end