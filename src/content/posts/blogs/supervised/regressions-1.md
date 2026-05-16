---
title: "Regression Models: Linear Regression and Regularization"
date: 2019/05/11
categories:
  - Blogs
tags:
  - Supervised Learning
  - Regression
excerpt: "A practical guide to linear regression, assumptions, loss, ordinary least squares, ridge, lasso, elastic net, and model evaluation."
layout: post
mathjax: true
toc: true
---

## Introduction

Linear regression models a numeric target as a weighted sum of input features:

$$
\hat{y} = w_0 + w_1x_1 + \dots + w_px_p
$$

It is simple, interpretable, and still useful as a baseline. Even when a more complex model wins, linear regression helps clarify the signal in the data.

## Ordinary Least Squares

Ordinary least squares chooses coefficients that minimize squared error:

$$
\sum_i (y_i - \hat{y}_i)^2
$$

Squared error penalizes large mistakes strongly.

## Assumptions

The classic assumptions include:

- Linear relationship between features and target.
- Independent errors.
- Constant error variance.
- No severe multicollinearity.
- Errors are approximately normal for inference.

Prediction can still work when assumptions are imperfect, but inference and interpretation become riskier.

## Regularization

Regularization discourages overly large coefficients and can reduce overfitting.

### Ridge Regression

Ridge adds an L2 penalty:

$$
\sum_i (y_i - \hat{y}_i)^2 + \lambda \sum_j w_j^2
$$

Ridge shrinks coefficients but usually does not make them exactly zero.

### Lasso Regression

Lasso adds an L1 penalty:

$$
\sum_i (y_i - \hat{y}_i)^2 + \lambda \sum_j |w_j|
$$

Lasso can set coefficients to zero, so it performs feature selection.

### Elastic Net

Elastic net combines L1 and L2 penalties. It is useful when features are correlated and lasso is unstable.

## Feature Scaling

Regularized regression usually needs feature scaling. Otherwise the penalty treats features differently because of units rather than importance.

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
model.fit(x_train, y_train)
```

## Evaluation

Use:

- MAE for interpretable average error.
- RMSE when large errors matter more.
- Residual plots for systematic patterns.
- Cross-validation when data is small.
- Slice evaluation when errors differ by group.

Linear regression is valuable because mistakes are often inspectable. Use that transparency before moving to a black-box model.
