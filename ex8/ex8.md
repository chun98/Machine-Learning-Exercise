# Anomaly Detection and Recommender Systems
## Anomaly detection
* estimateGaussian.m
  计算均值和方差
```matlab
mu = mean(X);
sigma2 = 1 / m * sum ( bsxfun(@minus, X, mu) .^2 );
```

* selectThreshold.m
  选择阈值
  * tp is the number of true positives: the ground truth label says it's an anomaly and our algorithm correctly classied it as an anomaly.
  * fp is the number of false positives: the ground truth label says it's not an anomaly, but our algorithm incorrectly classied it as an anomaly.
  * fn is the number of false negatives: the ground truth label says it's an anomaly, but our algorithm incorrectly classied it as not being anomalous.
  
    prec = tp / (tp + fp);      
    rec = tp / (tp + fn);       
    F1 = 2 * prec * rec / (prec + rec);     

```matlab
predictions = (pval < epsilon);
    
fp = sum((predictions == 1) & (yval == 0));
fn = sum((predictions == 0) & (yval == 1));
tp = sum((predictions == 1) & (yval == 1));
prec = tp / (tp + fp);
rec = tp / (tp + fn);
F1 = 2 * prec * rec / (prec + rec);
```

## Recommender Systems
* cofiCostFunc.m
  计算代价函数和梯度
  ![dca2a1ab2366bcf09d75cd6ba6812777.png](https://us1.myximage.com/2018/08/17/dca2a1ab2366bcf09d75cd6ba6812777.png)     
  ![7bd40508a3187f39ea34d918ea0b0fb3.png](https://us1.myximage.com/2018/08/17/7bd40508a3187f39ea34d918ea0b0fb3.png)
```matlab
J_temp = (X * Theta' - Y).^2;
J = sum(sum(J_temp(R == 1)))/2 + lambda/2 .* sum(sum(Theta.^2)) + lambda/2 .* sum(sum(X.^2));

X_grad = ((X * Theta' - Y) .* R) * Theta + lambda.*X;
Theta_grad = ((X * Theta' - Y) .* R)' * X + lambda.*Theta;
```