---
title: "Regression Models: GAM, GLM, and GLMM"
date: 2019/05/30
categories:
  - Blogs
tags:
  - Supervised Learning
  - Regression
excerpt: "A concise explanation of generalized linear models, generalized additive models, and generalized linear mixed models, with guidance on when to use each."
layout: post
mathjax: true
toc: true
---

## Introduction

Linear regression is a starting point, but many real targets are not continuous normal variables with simple linear effects. GLM, GAM, and GLMM extend regression in different directions.

## GLM: Generalized Linear Model

A generalized linear model connects features to the expected target through a link function.

It has three pieces:

- A linear predictor.
- A probability distribution for the target.
- A link function.

Examples:

- Logistic regression for binary outcomes.
- Poisson regression for counts.
- Gamma regression for positive continuous values.

Use GLMs when the target distribution is not well modeled by ordinary linear regression.

## GAM: Generalized Additive Model

A generalized additive model allows nonlinear feature effects while keeping the model additive:

$$
g(E[y]) = \beta_0 + f_1(x_1) + f_2(x_2) + \dots + f_p(x_p)
$$

Each $f_i$ can be a smooth function.

Use GAMs when:

- Nonlinear effects matter.
- Interpretability still matters.
- You want plots showing how each feature affects the prediction.

GAMs are a good middle ground between linear models and black-box models.

## GLMM: Generalized Linear Mixed Model

A generalized linear mixed model adds random effects.

Use GLMMs when data has grouped structure:

- Students within schools.
- Patients within hospitals.
- Transactions within users.
- Measurements within devices.

Random effects help model variation between groups without fitting a completely separate model for each group.

## Choosing Among Them

Use:

- **GLM** when the target distribution or link function matters.
- **GAM** when nonlinear feature effects matter and interpretability is important.
- **GLMM** when observations are grouped or repeated.

For pure predictive accuracy on tabular data, boosted trees may win. For inference and explainability, these regression families remain valuable.

## Closing

GLM, GAM, and GLMM extend linear regression while preserving statistical structure. They are especially useful when you need both prediction and understanding.
