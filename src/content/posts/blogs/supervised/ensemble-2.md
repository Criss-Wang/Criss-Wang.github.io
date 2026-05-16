---
title: "Ensemble Models: Boosting Techniques"
date: 2021/02/04
categories:
  - Blogs
tags:
  - Supervised Learning
  - Ensembles
excerpt: "A practical explanation of boosting, including AdaBoost, gradient boosting, XGBoost-style regularization, learning rate, tree depth, and common pitfalls."
layout: post
mathjax: true
toc: true
---

## Introduction

Boosting builds an ensemble sequentially. Each new learner focuses on what the current ensemble handles poorly.

Boosting can produce excellent performance, especially on tabular data, but it can overfit if the model is too flexible or trained too long.

## AdaBoost

AdaBoost trains weak learners in sequence and increases the weight of examples that were misclassified.

The final model is a weighted vote of weak learners.

AdaBoost is historically important and still useful for understanding the boosting idea, though gradient boosting is more common in modern tabular ML.

## Gradient Boosting

Gradient boosting builds an additive model:

$$
F_M(x) = \sum_{m=1}^{M} \eta f_m(x)
$$

Each new model fits the negative gradient of the loss with respect to the current prediction. For squared error, this is similar to fitting residuals.

The learning rate $\eta$ controls how much each new learner contributes.

## Tree-Based Boosting

Most practical boosting uses shallow decision trees as base learners.

Popular libraries:

- XGBoost.
- LightGBM.
- CatBoost.
- scikit-learn gradient boosting.

They add engineering improvements such as regularization, efficient split finding, missing-value handling, categorical support, and parallelization.

## Key Hyperparameters

Important settings:

- Number of trees.
- Learning rate.
- Maximum depth.
- Minimum child weight or samples per leaf.
- Subsampling rows.
- Subsampling columns.
- L1 or L2 regularization.
- Early stopping.

Learning rate and number of trees interact. A smaller learning rate usually needs more trees.

## Why Boosting Works

Boosting is strong because it gradually corrects errors. Early trees learn broad patterns. Later trees refine difficult cases.

This makes boosting powerful, but also risky. If allowed to chase noise, it can overfit.

## Practical Warnings

Watch for:

- Leakage in validation.
- Overfitting after too many trees.
- Bad calibration.
- Poor performance on rare slices.
- Slow inference with too many trees.
- Hyperparameter search overfitting.

Use early stopping with a validation set:

```python
model.fit(
    x_train,
    y_train,
    eval_set=[(x_valid, y_valid)],
)
```

Exact syntax depends on the library.

## Closing

Boosting is one of the most reliable tools for structured data. Treat it as a strong baseline, tune it carefully, and validate against both aggregate and slice metrics.
