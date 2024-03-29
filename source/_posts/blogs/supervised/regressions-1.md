---
title: "Regression Models: Linear Regression and Regularization"
excerpt: "Shallow and Deep Linear Regression"
layout: post
date: 2019/05/11
updated: 2021/6/14
categories:
  - Blogs
tags: 
  - Supervised Learning
  - Regression
  - Regularization
mathjax: true
toc: true
---
### Definition
<ul>
    <li>It is used for predicting the continuous dependent variable with the help of independent variables.</li>
    <li>The goal is to find the best fit line that can accurately predict the output for the continuous dependent variable.</li>
    <li>The model is usually fit by minimizing the sum of squared errors (<b>OLS (Ordinary Least Square)</b> estimator for regression parameters)</li>
    <li>Major algorithm is gradient descent: the key is to adjust the learning rate $\alpha$</li>
    <li>Explanation in layman terms: 
        <br>
        - provides you with a straight line that lets you infer the dependent variables
        <br> 
        - estimate the trend of a continuous data by a straight line. using input data to <b> predict </b>the outcome in the best possible way given the past data and its corresponding past outcomes </li>
</ul> 

### Various Regulations
Regularization is a simple techniques to reduce *model complexity* and prevent *over-fitting* which may result from simple linear regression.
<ul>
    <li>Convergence conditions differ</li>
    <li>note that regularization only apply on variables (hence $\theta_0$ is not regularized!)</li>
    <li>L2 norm:  Euclidean distance from the origin</li>
    <li>L1 norm:  Manhattan distance from the origin</li>
    <li>Elastic Net:  Mixing L1 and L2 norms</li>
    <li>Ridge regression: $\lambda  \sum_{k=1}^n (\theta_k)^2$ where $\theta_k$ is cofficient; more widely used as compared to Ridge when number of variables increases </li>
    <li>Lasso regression: $\lambda  \sum_{k=1}^n  |\theta_k|$; better when the data contains suspicious collinear variables</li>
</ul> 

### Comparison with Logistic Regression
- Linear Regression: the outcomes are continuous (infinite possible values); error minimization technique is ordinary least square.
- Logistic Regression: outcomes usually have limited number of possible values; error minimization technique is maximal likelihood. 

### Implementations
Basic operations using <kbd>sklearn</kbd> packages

```python
from sklearn.linear_model import LinearRegression
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])

y = np.dot(X, np.array([1, 2])) + 3
reg = LinearRegression(normalize=False, fit_intercept = True).fit(X, y)

display(reg.score(X, y))
display(reg.coef_) # regression coefficients
display(reg.intercept_) # y-intercept / offset

reg.predict(np.array([[3, 5]]))
```

### Common Questions

<ul>
    <li>Is Linear regression sensitive to outliers? <b>Yes!</b></li>
    <li>Is a relationship between residuals and predicted values in the model ideal? <b>No, residuals should be due to randomness, hence no relationship is an ideal property for th model</b></li>
    <li>What is the range of learning rate? <b>0 to 1 </b></li>
</ul> 

### Advanced: Analytical solutions
Here let\'s discuss some more math-intensive stuff. Those who are not interested can ignore this part (though it gives a very important guide on regression models)

#### 1. A detour into Hypothesis representation

We will use $\mathbf{x_i}$ to denote the independent variable and $\mathbf{y_i}$ to denote dependent variable. A pair of $\mathbf{(x_i,y_i)}$ is called training example. The subscripe $\mathbf{i}$ in the notation is simply index into the training set. We have $\mathbf{m}$ training example then $\mathbf{i = 1,2,3,...m}$.

The goal of supervised learning is to learn a *hypothesis function $\mathbf{h}$*, for a given training set that can used to estimate $\mathbf{y}$ based on $\mathbf{x}$. So hypothesis fuction represented as 

$$\mathbf{ h_\theta(x_{i}) = \theta_0 + \theta_1x_i }$$

where $\mathbf{\theta_0,\theta_1}$ are parameter of hypothesis.This is equation for **Simple / Univariate Linear regression**. 

For **Multiple Linear regression** more than one independent variable exit then we will use $\mathbf{x_{ij}}$ to denote indepedent variable and $\mathbf{y_{i}}$ to denote dependent variable. We have $\mathbf{n}$ independent variable then $\mathbf{j=1,2,3 ..... n}$. The hypothesis function represented as

$$\mathbf{h_\theta(x_{i}) = \theta_0 + \theta_1x_{i1} + \theta_2 x_{i2} + ..... \theta_j x_{ij} ...... \theta_n  x_{mn} }$$

where $\mathbf{\theta_0,\theta_1,....\theta_j....\theta_n }$ are parameter of hypothesis, $\mathbf{m}$ Number of training exaples, $\mathbf{n}$ Number of independent variable, $\mathbf{x_{ij}}$ is $\mathbf{i^{th}}$ training exaple of $\mathbf{j^{th}}$ feature.

#### 2. Matrix Formulation
In general we can write above vector as 

$$ \mathbf{ x_{ij}} = \left( \begin{smallmatrix} \mathbf{x_{i1}} & \mathbf{x_{i2}} &.&.&.& \mathbf{x_{in}} \end{smallmatrix} \right)$$

Now we combine all aviable individual vector into single input matrix of size $(m,n)$ and denoted it by $\mathbf{X}$ input matrix, which consist of all training exaples,

$$\mathbf{X} = \Bigg( \begin{smallmatrix} x_{11} & x_{12} &.&.&.&.& x_{1n}\\\\
                                x_{21} & x_{22} &.&.&.&.& x_{2n}\\\\
                                x_{31} & x_{32} &.&.&.&.& x_{3n}\\\\
                                .&.&.&. &.&.&.& \\\\
                                .&.&.&. &.&.&.& \\\\
                                x_{m1} & x_{m2} &.&.&.&.&. x_{mn}\\\\
                                \end{smallmatrix} \Bigg)_{(m,n)}$$

We represent parameter of function and dependent variable in vactor form as

$$\theta = \left(\begin{matrix} \theta_0 \\\\ \theta_1 \\\\ .\\\\.\\\\ \theta_j\\\\.\\\\.\\\\ \theta_n \end{matrix}\right)_{(n+1,1)} $$

$$\mathbf{ y } = \left (\begin{matrix} y_1\\\\ y_2\\\\. \\\\. \\\\ y_i \\\\. \\\\. \\\\ y_m \end{matrix} \right)_{(m,1)}$$

So we represent hypothesis function in vectorize form $\mathbf{ h_\theta{(x)} = X\theta}$.

#### 3. Cost function

A cost function measures how much error in the model is in terms of ability to estimate the relationship between $x$ and $y$. 
We can measure the accuracy of our hypothesis function by using a cost function. This takes an average difference of observed dependent variable in the given the dataset and those predicted by the hypothesis function.
  
$$\mathbf{ J(\theta) = \frac{1}{m} \sum_{i=1}^{m}(\hat{y}_i - y_i)^2}$$

$$\mathbf{J(\theta) =  \frac{1}{m} \sum_{i=1}^{m}(h_\theta(x_i) - y_i)^2}$$

To implement the linear regression, take training example add an extra column that is $x_0$ feature, where $\mathbf{x_0=1}$. $\mathbf{x_{o}} = ( \begin{smallmatrix} x_{i0} & x_{i1} & x_{i2} &.&.&.& x_{mi} \end{smallmatrix} )$,where $\mathbf{x_{i0} =0}$ and input matrix will become as

$$\mathbf{X} = \left( \begin{smallmatrix} x_{10} & x_{11} & x_{12} &.&.&.&.& x_{1n}\\\\
                                x_{20} & x_{21} & x_{22} &.&.&.&.& x_{2n}\\\\
                                x_{30} & x_{31} & x_{32} &.&.&.&.& x_{3n}\\\\
                                 .&.&.&.&. &.&.&.& \\\\
                                 .&.&.&.&. &.&.&.& \\\\
                                x_{m0} & x_{m1} & x_{m2} &.&.&.&.&. x_{mn}\\\\
                                \end{smallmatrix} \right)_{(m,n+1)}$$ 

Each of the m input samples is similarly a column vector with n+1 rows $x_0$ being 1 for our convenience, that is $\mathbf{x_{10},x_{20},x_{30} .... x_{m0} =1}$. Now we rewrite the ordinary least square cost function in matrix form as

$$\mathbf{J(\theta) = \frac{1}{m} (X\theta - y)^T(X\theta - y)}$$

Let\'s look at the matrix multiplication concept,the multiplication of two matrix happens only if number of column of firt matrix is equal to number of row of second matrix. Here input matrix $\mathbf{X}$ of size $\mathbf{(m,n+1)}$, parameter of function is of size $(n+1,1)$ and dependent variable vector of size $\mathbf{(m,1)}$. The product of matrix $\mathbf{X_{(m,n+1)}\theta_{(n+1,1)}}$ will return a vector of size $\mathbf{(m,1)}$, then product of $\mathbf{(X\theta - y)^T_{(1,m)}(X\theta - y)_{(m,1)}}$ will return size of unit vector. 

#### 4. Normal Equation
The normal equation is an analytical solution to the linear regression problem with a ordinary least square cost function. To minimize our cost function, take partial derivative of $\mathbf{J(\theta)}$ with respect to $\theta$ and equate to $0$. The derivative of function is nothing but if a small change in input what would be the change in output of function.

$$\mathbf{\min_{\theta_0,\theta_1..\theta_n} J({\theta_0,\theta_1..\theta_n})}\\\\
\mathbf{\frac{\partial J(\theta_j)}{\partial\theta_j} =0}$$

where $\mathbf{j = 0,1,2,....n}$

Now we will apply partial derivative of our cost function,

$$\mathbf{\frac{\partial J(\theta_j)}{\partial\theta_j} = \frac{\partial }{\partial \theta} \frac{1}{m}(X\theta - y)^T(X\theta - y) }$$

I will throw $\mathbf{\frac {1}{m}}$ part away since we are going to compare a derivative to $0$. And solve $\mathbf{J(\theta)}$,  

$$ \begin{align}\mathbf{J(\theta)}
&\mathbf{= (X\theta -y)^T(X\theta - y)}\\\\
&\mathbf{= (X\theta)^T - y^T)(X\theta -y)}  \\\\
&\mathbf{= (\theta^T X^T - y^T)(X\theta - y)} \\\\
&\mathbf{= \theta^T X^T X \theta - y^T X \theta - \theta^T X^T y + y^T y} \\\\
&\mathbf{ = \theta^T X^T X \theta  - 2\theta^T X^T y + y^T y} \end{align}$$

Here $\mathbf{y^T_{(1,m)} X_{(m,n+1)} \theta_{(n+1,1)} = \theta^T_{(1,n+1)} X^T_{(n+1,m)} y_{(m,1)}}$ because of unit vector.

$$\mathbf{\frac{\partial J(\theta)}{\partial \theta} = \frac{\partial}{\partial \theta} (\theta^T X^T X \theta  - 2\theta^T X^T y + y^T y )} \\\\
\mathbf{ = X^T X \frac {\partial \theta^T \theta}{\partial\theta} - 2 X^T y \frac{\partial \theta^T}{\partial\theta} + \frac {\partial y^T y}{\partial\theta}}$$

Partial derivative $\mathbf{\frac {\partial x^2}{\partial x} = 2x}$, $\mathbf{\frac {\partial kx^2}{\partial x} = kx}$,$\mathbf{\frac {\partial Constact}{\partial x} = 0}$, hence 

$$\begin{align}\mathbf{\frac{\partial J(\theta)}{\partial\theta}\\;} &\mathbf{= X^T X 2\theta - 2X^T y +0} \\\\
\mathbf{ 0 \\;} &\mathbf{= 2X^T X \theta - 2X^T y} \\\\
\mathbf{ X^T X \theta \\;} &\mathbf{= X^T } \\\\
\mathbf{ \theta \\;} &\mathbf{= (X^TX)^{-1} X^Ty }\end{align}$$

this $\mathbf{ \theta = (X^TX)^{-1} X^Ty }$ is the normal equation for linear regression.
### Advanced: Model Evaluation and Model Validation

#### 1. Model evaluation
We will predict value for target variable by using our model parameter for test data set. Then compare the predicted value with actual valu in test set. We compute **Mean Square Error** using formula 

$$\mathbf{ J(\theta) = \frac{1}{m} \sum_{i=1}^{m}(\hat{y}_i - y_i)^2}$$

$\mathbf{R^2}$ is statistical measure of how close data are to the fitted regression line. $\mathbf{R^2}$ is always between 0 to 100%. 0% indicated that model explains none of the variability of the response data around it\'s mean. 100% indicated that model explains all the variablity of the response data around the mean.

$$\mathbf{R^2 = 1 - \frac{SS_E}{SS_T}}$$

where $SS_E$ = Sum of Square Error, $SS_T$ = Sum of Square Total.

$$ \mathbf{SS_E = \sum\limits_{i=1}^{m}(\hat{y}_i - y_i)^2} $$

$$\mathbf{SS_T = \sum\limits_{i=1}^{m}(y_i - \bar{y}_i)^2}$$


Here $\mathbf{\hat{y}}$ is predicted value and $\mathbf{\bar{y}}$ is mean value of $\mathbf{y}$.
Below is a sample code for evaluation

```python
# Normal equation
y_pred_norm =  np.matmul(X_test_0,theta)

#Evaluvation: MSE
J_mse = np.sum((y_pred_norm - y_test)**2)/ X_test_0.shape[0]

# R_square 
sse = np.sum((y_pred_norm - y_test)**2)
sst = np.sum((y_test - y_test.mean())**2)
R_square = 1 - (sse/sst)
print('The Mean Square Error(MSE) or J(theta) is: ',J_mse)
print('R square obtain for normal equation method is :',R_square)
>>> The Mean Square Error(MSE) or J(theta) is:  0.17776161210877062
>>> R square obtain for normal equation method is : 0.7886774197617128

# sklearn regression module
y_pred_sk = lin_reg.predict(X_test)

#Evaluvation: MSE
from sklearn.metrics import mean_squared_error
J_mse_sk = mean_squared_error(y_pred_sk, y_test)

# R_square
R_square_sk = lin_reg.score(X_test,y_test)
print('The Mean Square Error(MSE) or J(theta) is: ',J_mse_sk)
print('R square obtain for scikit learn library is :',R_square_sk)
>>> The Mean Square Error(MSE) or J(theta) is:  0.17776161210877925
>>> R square obtain for scikit learn library is : 0.7886774197617026
```
The model returns $R^2$ value of 77.95%, so it fit our data test very well, but still we can imporve the the performance of by diffirent technique. Please make a note that we have transformer out variable by applying  natural log. When we put model into production antilog is applied to the equation.

#### 2. Model Validation
In order to validated model we need to check few assumption of linear regression model. The common assumption for *Linear Regression* model are following
1. Linear Relationship: In linear regression the relationship between the dependent and independent variable to be *linear*. This can be checked by scatter ploting Actual value Vs Predicted value
2. The residual error plot should be *normally* distributed.
3. The *mean* of *residual error* should be 0 or close to 0 as much as possible
4. The linear regression require all variables to be multivariate normal. This assumption can best checked with Q-Q plot.
5. Linear regession assumes that there is little or no *Multicollinearity in the data. Multicollinearity occurs when the independent variables are too highly correlated with each other. The variance inflation factor *VIF* identifies correlation between independent variables and strength of that correlation. $\mathbf{VIF = \frac {1}{1-R^2}}$, If VIF >1 & VIF <5 moderate correlation, VIF < 5 critical level of multicollinearity.
6. Homoscedasticity: The data are homoscedastic meaning the residuals are equal across the regression line. We can look at residual Vs fitted value scatter plot. If heteroscedastic plot would exhibit a funnel shape pattern.

The model assumption linear regression as follows
1. In our model  the actual vs predicted plot is curve so linear assumption fails
2. The residual mean is zero and residual error plot right skewed
3. Q-Q plot shows as value log value greater than 1.5 trends to increase
4. The plot is exhibit heteroscedastic, error will insease after certian point.
5. Variance inflation factor value is less than 5, so no multicollearity.

<figure align="center">
  <img src="/images/Machine%20learning/regression_plot-1.png" width="500px">
  <figcaption>Linearity plot and Residual plot.</figcaption>
</figure>
<figure align="center">
  <img src="/images/Machine%20learning/regression_plot-2.png" width="500px">
  <figcaption>Q-Q Plot and HomoScedasticity plot</figcaption>
</figure>


