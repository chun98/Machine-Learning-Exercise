# Logistic Regression

* plotData.m    
  绘图
```matlab
% Find Indices of Positive and Negative Examples
pos = find(y==1); neg = find(y == 0);
% Plot Examples
plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2, ...
'MarkerSize', 7);
plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', ...
'MarkerSize', 7);
```

* sigmoid.m  
  计算sigmoid函数
```matlab
g = zeros(size(z));
g = 1 ./ ( 1 + exp(-z) ) ;
```
* costFunction.m    
  计算代价函数
```matlab
J= -1 * sum( y .* log( sigmoid(X*theta) ) + (1 - y ) .* log( (1 - sigmoid(X*theta)) ) ) / m ;
grad = ( X' * (sigmoid(X*theta) - y ) )/ m ;
```

* ex2.m     
  用fminunc求θ最优解
```matlab
options = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);
```

* predict.m     
  预测
```matlab
k = find(sigmoid( X * theta) >= 0.5 );
p(k)= 1;
```

# Regularized logistic regression
* costFunctionReg.m
```matlab
theta_1=[0;theta(2:end)];    % 先把theta(1)拿掉，不参与正则化
J= -1 * sum( y .* log( sigmoid(X*theta) ) + (1 - y ) .* log( (1 - sigmoid(X*theta)) ) ) / m  + lambda/(2*m) * theta_1' * theta_1 ;
grad = ( X' * (sigmoid(X*theta) - y ) )/ m + lambda/m * theta_1 ;
```