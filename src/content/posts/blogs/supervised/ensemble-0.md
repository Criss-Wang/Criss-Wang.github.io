---
title: "Ensemble Models: Overview"
date: 2021/01/18
categories:
  - Blogs
tags:
  - Supervised Learning
  - Ensembles
excerpt: "A practical overview of ensemble learning, including bagging, boosting, stacking, voting, variance reduction, bias reduction, and common tradeoffs."
layout: post
mathjax: true
toc: true
---

## Introduction

Ensemble learning combines multiple models to produce a stronger predictor than a single model. The idea is simple: different models make different errors, and combining them can reduce the overall error.

The main families are:

- Voting.
- Bagging.
- Boosting.
- Stacking.

## Why Ensembles Work

Prediction error can be thought of in terms of bias, variance, and noise.

- High bias: the model is too simple.
- High variance: the model changes too much with the training data.
- Noise: irreducible randomness or measurement error.

Bagging mainly reduces variance. Boosting often reduces bias and can reduce variance when regularized well.

## Voting

Voting combines predictions from several models.

- **Hard voting:** choose the class with the most votes.
- **Soft voting:** average predicted probabilities and choose the largest.

Voting is simple and useful when models are diverse and individually reasonable.

## Bagging

Bagging trains many models on bootstrap samples of the data, then averages or votes across predictions.

Random forests are the most common example. They train many decision trees on sampled data and sampled features.

Bagging is useful when the base learner has high variance, such as decision trees.

## Boosting

Boosting trains models sequentially. Each new model focuses on mistakes made by previous models.

Examples:

- AdaBoost.
- Gradient Boosting.
- XGBoost.
- LightGBM.
- CatBoost.

Boosting is often very strong on tabular data, but it needs careful regularization.

## Stacking

Stacking trains multiple base models, then trains a meta-model to combine their predictions.

The meta-model must be trained on out-of-fold predictions to avoid leakage. If it sees predictions from models trained on the same labels, validation scores can be too optimistic.

## Practical Tradeoffs

Ensembles often improve accuracy but add complexity:

- More training cost.
- More inference cost.
- More difficult debugging.
- More complicated deployment.
- Less interpretability.

Use ensembles when the metric improvement is worth the operational cost.

## Closing

Ensembles are powerful because they turn model diversity into robustness. Start with a strong single model, then use bagging, boosting, or stacking when the problem justifies the extra complexity.
