# Support Vector Machines

* gaussianKernel.m      
  高斯核函数        
  ![89a44f0146176efa131942d0d832a3a6.png](https://us1.myximage.com/2018/08/12/89a44f0146176efa131942d0d832a3a6.png)
```matlab
sim = exp( - (x1-x2)'* (x1-x2) / (2 * sigma *sigma ) );
```

* dataset3Params.m      
  选出最优的C和σ。
```matlab
C_vec = [0.01 0.03 0.1 0.3 1 3 10 30]';
sigma_vec = [0.01 0.03 0.1 0.3 1 3 10 30]';
error_val = zeros(length(C_vec),length(sigma_vec));
error_train = zeros(length(C_vec),length(sigma_vec));
for i = 1:length(C_vec)
    for j = 1:length(sigma_vec)
      model= svmTrain(X, y, C_vec(i), @(x1, x2) gaussianKernel(x1, x2, sigma_vec(j))); 
      predictions = svmPredict(model, Xval);
      error_val(i,j) = mean(double(predictions ~= yval));
    end
end

[minval,ind] = min(error_val(:));   
[I,J] = ind2sub([size(error_val,1) size(error_val,2)],ind);
C = C_vec(I)         
sigma = sigma_vec(J) 
```

* processEmail.m        
  在vocabList查找，如果查找到，就加入word_indices中。
```matlab
for i=1:length(vocabList)
    if( strcmp(vocabList{i}, str) )
      word_indices = [word_indices;i];
    end
end
```

* emailFeatures.m
  根据word_indices产生一个特征向量.
```matlab
x(word_indices) = 1;
```