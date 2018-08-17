# Multi-class Classication and Neural Networks

* lrCostFunction        
  计算代价函数      
  ![75266998c936f96c07d8bccef8df4bca.png](https://us1.myximage.com/2018/08/10/75266998c936f96c07d8bccef8df4bca.png)
```matlab
temp=[0;theta(2:end)];    % 先把theta(1)拿掉，不参与正则化
J= -1 * sum( y .* log( sigmoid(X*theta) ) + (1 - y ) .* log( (1 - sigmoid(X*theta)) ) ) / m  + lambda/(2*m) * temp' * temp ;
grad = ( X' * (sigmoid(X*theta) - y ) )/ m + lambda/m * temp ;
```

* oneVsAll.m      
  多类别分类
```matlab
options = optimset('GradObj', 'on', 'MaxIter', 50);
initial_theta = zeros(n + 1, 1);
for c = 1:num_labels
all_theta(c,:) = fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
            initial_theta, options);
end
```

* predictOneVsAll.m       
  多类别预测
```matlab
[a,p] = max(sigmoid( X * all_theta'),[],2) ;    % 返回每行最大值的索引位置，也就是预测的数字
```

* predict.m     
  神经网络结构      
  ![ab29f0b0b1f166b524dca928b01fcc37.png](https://us1.myximage.com/2018/08/10/ab29f0b0b1f166b524dca928b01fcc37.png)
```matlab
X = [ones(m, 1) X];
a2 = sigmoid(X * Theta1');   % 第二层激活函数输出
a2 = [ones(m, 1) a2];        % 第二层加入b
a3 = sigmoid(a2 * Theta2');  
[aa,p] = max(a3,[],2);  % 返回每行最大值的索引位置，也就是预测的数字
```