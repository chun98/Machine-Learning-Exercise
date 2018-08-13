# K-means Clustering and Principal Component Analysis

* findClosestCentroids.m        
  找到距离最近的聚类中心，并标上号。
```matlab
for i=1:length(idx)
    distanse = pdist2(centroids,X(i,:));   
    [C,idx(i)]=min(distanse);           % find the minimum
end
```
* computeCentroids.m        
  计算聚类中心的点的平均值
```matlab
for i=1:K
       centroids(i,:) =  mean( X( find(idx==i) , :) );   % 
end
```
* kMeansInitCentroids.m     
  初始化聚类中心
```matlab
randidx = randperm(size(X, 1));
centroids = X(randidx(1:K), :);
```
       
* pca.m     
  计算协方差矩阵
  ![3c20755ca8c6dcdb040b82b59d8be709.png](https://us1.myximage.com/2018/08/12/3c20755ca8c6dcdb040b82b59d8be709.png)
```matlab
  Sigma = 1/m * X'* X;
  [U, S, V] = svd(Sigma);
```
* projectData.m     
  获取U_reduce，计算新特征向量Z。
```matlab
     U_reduce = U(:, 1:K);
     Z =X * U_reduce;
```
* reciverData.m
  恢复原有的高维数据
```matlab
U_reduce = U(:, 1:K);

X_rec = Z * U_reduce';
```
