---
title: "Regression Models: GAM, GLM and GLMM"
date: 2019-05-16
layout: single
author_profile: true
categories:
  - Regression
tags: 
  - Linear Model
  - Estimation
excerpt: "A brief introduction to generalized regression"
mathjax: "true"
---
## Overview
Generalized linear model (GLM) is a cure to some issues posted by ordinary linear regression. In the well-known linear regression model, we often assume $y=Xc+b$. However, it often assumes that $y$ is not bounded when $x$ is not bounded. However, very often, we must restrict the values of $y$ within a fixed range. This may invalidated the ordinary linear model as the function behaviors near the boundary points can be very off. Generalized linear models aim to deal with this issue by allowing for $y$ that have arbitrary distributions (not just gaussian distribution), a function of $y$ (the link function) to vary linearly with $x$ (rather than assuming that a direct linear relationship between $x$ and $y$). Generalized Additive Model (GAM) and Generalized Linear Mixed Model (GLMM) are extensions to GLM with special functions applied to differenet elements in $x$. 
## GLM
We note that GLM has three major parts:
1. An exponential family of probability distributions: $f(x \|\theta) = h(x)\text{exp}[\eta (\theta)) \cdot T(x) + A(\theta)]$, some examples include:
   - normal
   - exponential
   - gamma
   - chi-squared
   - beta
   - Dirichlet
   - Bernoulli
   - categorical
   - Poisson
2. A function of predictor (in GLM it is $Xc$, in extended models, it can be other things, see GAM and GLMM), we can estimate $c$ via *maximum likelihood* or Bayesian methods like *laplace approximation* and *Gibbs sampling*, etc.
3. A link function $g$ such that $g(E(Y\|X)) = Xc$ (sometime we may have tractable distribution for variance $Var(Y\|X) = V(g^{-1}(Xc)))$

### Pros and Cons for GLM and GLMM

- Pros:
    - Easy to interpret
    - Easy to grasp
    - Coefficients can be further used in numerical models
    - Easy to extend: link functions, fixed and random effects, correlation structures
    
- Cons:
    - Not good for dynamic models (the model is not linear and transformation may not help or would loose information

## Generalized additive models (GAMs)
- GAMs are extensions to GLMs in which the linear predictor $\eta = Xc$ is not restricted to be linear in the covariates $X$ but is the sum of smoothing functions applied to the each $x_i$. For example,

$g(E(Y\|X)) = c_0 + f_1(x_1) + ... + f_n(x_n)$.
{: style="text-align: center;"}
- Is useful if relationship between Y and X is likely to be non-linear but we **don't have any theory or any mechanistic model** to suggest a particular functional form
- Each $Y_i$ is linked with $X_i$ by a **smoothing function** instead of a coefficient $\beta$
- GAMS are **data-driven** rather than model-driven, that is, the resulting fitted values do not come from an a priori model (non-parametric)
- **All of the distribution families** allowed with GLM are available with GAM

### Pros and Cons for GAM

- __Pros__: 
    - By combining the basis functions GAMs can represent a large number of functional relationship (to do so they rely on the assumption that the true relationship is likely to be smooth, rather than wiggly)
    - Particularly useful for uncovering nonlinear effects of numerical covariates, and for doing so in an "automatic" fashion
    - More Flexible as now each sample's Y is associated with its X by a smoothing function instead of a coefficient $\beta$
- __Cons__:
    - Interpretability of the coefficient -> need to be estimated graphically
    - Coefficients are not easily transferable to other datasets and parameterization
    - Very sensitive to gaps in the data and outliers
    - Lack underlying theory for the use of hypothesis tests -> one solution is to do bootstrapping and get aggregated result for more reliable confidence bands

### Examples of GAM (different predictor representation functions):
1. Loess (Locally weighted regression smoothing)
    - The key factor is the **span width** (usually set to be a proportion of the data set: 0.5 as a standard starting point)
    - Main idea: Split the data into separate blobs using sliding windows and fit linear regressions in each blob/interval
    - Pros:
        - Easily interpretable. At each test case, a local linear model is fit (eventually explained by linear behaviours)
        - a popular way to see smooth trends on scatterplots
    - Cons:
        - If there are a lot of data points, fitting a LOESS over the entire range of the predictor can be slow because so many local linear regressions must be fit.
    
    
2. Regression Splines (piecewise polynomials over usually a finite range)
    - Main constraint is that the splines must remain smooth and continuous at knots
    - To avoid overfitting of splines, penalty terms are added
    - The penalty term also reflects the **degree of smoothness** in the regression
    - The less smooth the regression is (after fitting the spline functions), the higher the penalty terms
    - Pros:
        - **cover all sorts of nonlinear trends** and are **computationally very attractive** because spline terms fit exactly into a least squares linear regression framework. Least squares models are very easy to fit computationally
    - Cons:
        - It is possible to create multidimensional splines by creating interactions between spline terms for different predictors. This suffers from the **curse of dimensionality** like KNN because we are trying to **estimate a wavy surface in a large dimensional (many variable) space where data points will only sparsely cover the many many regions of the space**

## GLMM
The model has the form: $g(E(Y\|X)) = Xc + Zu$ where $Z$ is the design matrix for the $q$ random effects (the random complement to the fixed $X$). $u$ is a vector of the random effects (the random complement to the fixed $c$). The random effects are just deviations around the value in $\beta$, which is the mean. Usually $Z$ is a sparse matrix that assigns random effects to each element. We nearly always assume that $u \sim {\cal N}(0, G)$ with $G$ being the covariance matrix of the random effects. Assuming that the random effects are independent, we can have $G$ being a diagonal matrix with entries $\sigma_{init}^2$ and $\sigma_{slope}^2$.  

### Code implementation
I recommend beginners to use <kbd>statsmodels</kbd> package because the output via `.summary()` function is very clear to read. For advanced users, you may implement the function yourself by referring to the mathematical expressions and package documentations from the following
- <kbd>statsmodels</kbd>: statsmodels.formula.api.mixedlm
- <kbd>pymc3</kbd>
- <kbd>theano</kbd>
- <kbd>pystan</kbd>
- <kbd>tensorflow</kbd>
- <kbd>keras</kbd>

#### A sample code using <kbd>statsmodels</kbd>

```python
import statsmodels.formula.api as smf
from patsy import dmatrices
formula = "rt ~ group*orientation*identity"
#formula = "rt ~ -1 + cbcond"
md  = smf.mixedlm(formula, tbltest, groups=tbltest["subj"])
mdf = md.fit()
print(mdf.summary())

fe_params = pd.DataFrame(mdf.fe_params,columns=['LMM'])
random_effects = pd.DataFrame(mdf.random_effects)
random_effects = random_effects.transpose()
random_effects = random_effects.rename(index=str, columns={'groups': 'LMM'})

#%% Generate Design Matrix for later use
Y, X   = dmatrices(formula, data=tbltest, return_type='matrix')
Terms  = X.design_info.column_names
_, Z   = dmatrices('rt ~ -1+subj', data=tbltest, return_type='matrix')
X      = np.asarray(X) # fixed effect
Z      = np.asarray(Z) # mixed effect
Y      = np.asarray(Y).flatten()
nfixed = np.shape(X)
nrandm = np.shape(Z)
```
        