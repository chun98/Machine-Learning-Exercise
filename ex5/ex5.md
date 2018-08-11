* linearRegCostFunction.m       
  计算代价函数和梯度      
  ![c22f707190fa58a6e60b840bd55e6717.png](https://us1.myximage.com/2018/08/11/c22f707190fa58a6e60b840bd55e6717.png)

```matlab
theta_1=[0;theta(2:end)];    % 先把theta(1)拿掉，不参与正则化
J = sum((X * theta - y).^2) / (2*m) + lambda/(2*m) * theta_1' * theta_1 ;    
grad = ( X' * (X * theta - y) )/ m + lambda/m * theta_1 ;
```
* learningCurve.m       
  产生学习速率

```matlab
 for i = 1:m   
      theta = trainLinearReg(X(1:i, :), y(1:i),lambda);
      error_train(i) = linearRegCostFunction(X(1:i, :), y(1:i),theta,0);
      error_val(i) = linearRegCostFunction(Xval, yval,theta,0);     
end
```

* polyFeatures.m        
  X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];       
```matlab
for i = 1 : p
    X_poly(:,i) = X.^i;
end
```
* validationCurve.m     
  回归分析
```matlab
for i = 1:length(lambda_vec)
      lambda = lambda_vec(i);
      theta = trainLinearReg(X, y,lambda);  % 用训练集训练出参数
      error_train(i) = linearRegCostFunction(X, y,theta,0);
      error_val(i) = linearRegCostFunction(Xval, yval,theta,0);           
end
```