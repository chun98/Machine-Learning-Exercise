# Exercise 1: Linear Regression

* warmUpExercise.m  
返回一个5X5的单位矩阵：
```matlab
A=eye(5);
```

* plotData.m    
绘制图像:
```matlab
figure; % open a new figure window
plot(x, y, 'rx', 'MarkerSize', 10); % Plot the data
ylabel('Profit in $10,000s'); % Set the y−axis label 
xlabel('Population of City in 10,000s'); % Set the x−axis label
```

* computeCost.m     
计算损失函数：
```matlab
m = length(y); 
J = 0;
J = X*theta-y;
J = (1/(2*m))*sum(J.*J);
```

* gradientDescent.m     
梯度下降算法：
```matlab
m = length(y); 
J_history = zeros(num_iters, 1);
J=zeros(m,1);
for iter = 1:num_iters
    J = X*theta-y;
    temp1=theta(1)-alpha*(1/m)*sum(J.*X(:,1));
    temp2=theta(2)-alpha*(1/m)*sum(J.*X(:,2));
    theta(1)=temp1;
    theta(2)=temp2;  
    J_history(iter) = computeCost(X, y, theta);
end
```
* computeCostMulti.m    
多变量的损失函数计算：
```matlab
m = length(y); 
J = 0;
J = sum((X * theta - y).^2) / (2*m);   
```
* gradientDescentMulti.m    
多变量的梯度下降算法：
```matlab
m = length(y); 
J_history = zeros(num_iters, 1);
for iter = 1:num_iters
    theta = theta - alpha / m * X' * (X * theta - y);   
    J_history(iter) = computeCostMulti(X, y, theta);
end
```
*  normalEqn.m  
正规方程解法：
```matlab
theta = zeros(size(X, 2), 1);
theta = pinv( X' * X ) * X' * y;
```