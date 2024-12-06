Xy = [Xtrain ytrain]
Xy1 = Xy(Xy(:,3)==1,:)
Xy2 = Xy(Xy(:,3)==2,:)
Xy3 = Xy(Xy(:,3)==3,:)

% 2.5a)
%gscatter(Xtrain(:,1),Xtrain(:,2),ytrain,clf)
gscatter(Xy1(:,1),Xy1(:,2),Xy1(:,3),"r")
hold on
gscatter(Xy2(:,1),Xy2(:,2),Xy2(:,3),"g")
gscatter(Xy3(:,1),Xy3(:,2),Xy3(:,3),"b")
legend("1","2","3")
xlabel('X') 
ylabel('Y') 
title('XY Scatterplot')
hold off


% 2.5b) contour graph for label 2
A = 6 : -0.1 : -3.5 %x
B = 6.5 : -0.1 : -3 %y
[A,B] = meshgrid(A,B)
output = zeros(size(A,1));
for i = 1:size(A,1)
    Xtest = [ A(:,i) B(:,i) ];
    output(:,i) =  knnProb(Xtest, Xy, 10, 2);   
end
contourf(A,B,output,11)
colorbar
xlabel('X') 
ylabel('Y') 
title('Heatmap of P(label=2)')

% 2.5b) contour graph for label 3
output = zeros(size(A,1));
for i = 1:size(A,1)
    Xtest = [ A(:,i) B(:,i) ];
    output(:,i) =  knnProb(Xtest, Xy, 10, 3);   
end
contourf(A,B,output,11)
colorbar
xlabel('X') 
ylabel('Y') 
title('Heatmap of P(label=3)')


% 2.5c) for [A,B] predict for k = 1, k = 5, use g1=r, g2=g, g3=b
% k = 1
%set rgb
mymap = [1.0 0 0; 0 1.0 0; 0 0 1.0]
labels = zeros(size(A,1));
for i = 1:size(A,1)
    Xtest = [ A(:,i) B(:,i) ];
    labels(:,i) =  knn(Xtest, Xy, 1);   
end
imagesc([6,-3.5],[6.5,-3],labels)
set(gca,'YDir','normal')
colormap(mymap)
xlabel('X') 
ylabel('Y')
title('Knn = 1')
colorbar

% k = 5
labels = zeros(size(A,1));
for i = 1:size(A,1)
    Xtest = [ A(:,i) B(:,i) ];
    labels(:,i) =  knn(Xtest, Xy, 5);   
end
imagesc([6,-3.5],[6.5,-3],labels)
set(gca,'YDir','normal')
colormap(mymap)
xlabel('X') 
ylabel('Y')
title('Knn = 5')
colorbar

%2.5d) Perform LOOCV on the training set and plot the average LOOCV CCR (k)
% where k = 1, 3, 5, 7, 9, 11
errork1 = LOOCV(Xy,1)
CCRk1 = 1 - errork1
errork3 = LOOCV(Xy,3)
CCRk3 = 1 - errork3
errork5 = LOOCV(Xy,5)
CCRk5 = 1 - errork5
errork7 = LOOCV(Xy,7)
CCRk7 = 1 - errork7
errork9 = LOOCV(Xy,9)
CCRk9 = 1 - errork9
errork11 = LOOCV(Xy,11)
CCRk11 = 1 - errork11
errork13 = LOOCV(Xy,13)
CCRk13= 1 - errork13
errork15 = LOOCV(Xy,15)
CCRk15 = 1 - errork15

%plot values k = 1 3 5 7 9 11
plot([1 3 5 7 9 11],[CCRk1 CCRk3 CCRk5 CCRk7 CCRk9 CCRk11],'--or')
xlabel('k') 
ylabel('CCR')
title('CCR(k)')

%plot values k = 1 3 5 7 9 11 13 15
plot([1 3 5 7 9 11 13 15],[CCRk1 CCRk3 CCRk5 CCRk7 CCRk9 CCRk11 CCRk13 CCRk15],'--or')
xlabel('k') 
ylabel('CCR')
title('CCR(k)')

%% 
% 

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
% takes point, an array of data+labels
% outputs an array of distance from arr to the point 
% as the next column in arr
    d = sqrt( power(arr(:,1) - point(1),2) + power(arr(:,2) - point(2),2));
    dist = [ arr d];
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