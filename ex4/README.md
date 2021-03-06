# Neural Networks Learning

* sigmoidGradient.m     
  计算sigmoid函数的梯度
```matlab
g = sigmoid(z) .* (1 - sigmoid(z));
```

* randInitializeWeights.m
```matlab
epsilon_init = 0.12;
W = rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init;
```

* nnCostFunction.m
```matlab
%% 对y进行处理 Y(find(y==3))= [0 0 1 0 0 0 0 0 0 0]; 用于 Feedforward cost function 1和2   
Y=[];
E = eye(num_labels);    % 要满足K可以是任意，则不能写eye(10)！！
for i=1:num_labels
    Y0 = find(y==i);    % 找到等于y=i的序列号,替换向量
    Y(Y0,:) = repmat(E(i,:),size(Y0,1),1);
end
% 计算前向传输 Add ones to the X data matrix  
X = [ones(m, 1) X];
a2 = sigmoid(X * Theta1');   % 第二层激活函数输出
a2 = [ones(m, 1) a2];        % 第二层加入b
a3 = sigmoid(a2 * Theta2');  

temp1 = [zeros(size(Theta1,1),1) Theta1(:,2:end)];   % 先把theta(1)拿掉，不参与正则化
temp2 = [zeros(size(Theta2,1),1) Theta2(:,2:end)];
temp1 = sum(temp1 .^2);     % 计算每个参数的平方，再就求和
temp2 = sum(temp2 .^2);

cost = Y .* log(a3) + (1 - Y ) .* log( (1 - a3));  % cost是m*K(5000*10)的结果矩阵  sum(cost(:))全部求和
J= -1 / m * sum(cost(:)) + lambda/(2*m) * ( sum(temp1(:))+ sum(temp2(:)) );  


%% 计算 Gradient 
delta_1 = zeros(size(Theta1));
delta_2 = zeros(size(Theta2));

for t = 1:m
   % step 1
   a_1 = X(t,:)';
   z_2 = Theta1 * a_1;   
   a_2 = sigmoid(z_2);  
   a_2 = [1 ; a_2];
   z_3 = Theta2 * a_2;
   a_3 = sigmoid(z_3);
   % step 2
   err_3 = zeros(num_labels,1);
   for k = 1:num_labels     
      err_3(k) = a_3(k) - (y(t) == k);
   end
   % step 3
   err_2 = Theta2' * err_3;                % err_2有26行！！！
   err_2 = err_2(2:end) .* sigmoidGradient(z_2);   % 去掉第一个误差值，减少为25. sigmoidGradient(z_2)只有25行！！！
   % step 4
   delta_2 = delta_2 + err_3 * a_2';
   delta_1 = delta_1 + err_2 * a_1';
end

% step 5
Theta1_temp = [zeros(size(Theta1,1),1) Theta1(:,2:end)];
Theta2_temp = [zeros(size(Theta2,1),1) Theta2(:,2:end)];
Theta1_grad = 1 / m * delta_1 + lambda/m * Theta1_temp;
Theta2_grad = 1 / m * delta_2 + lambda/m * Theta2_temp ;
```