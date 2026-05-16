---
title: "Regression Models: Logistic Regression"
date: 2019/05/13
categories:
  - Blogs
tags:
  - Supervised Learning
  - Classification
excerpt: "A practical introduction to logistic regression for binary classification, odds, probabilities, regularization, thresholds, and evaluation."
layout: post
mathjax: true
toc: true
---

## Introduction

Despite the name, logistic regression is usually used for classification. It models the probability of a binary outcome.

For features $x$, logistic regression predicts:

$$
p(y=1 \mid x) = \sigma(w^Tx + b)
$$

where:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

## Interpretation

The model is linear in log-odds:

$$
\log \frac{p}{1-p} = w^Tx + b
$$

This makes logistic regression more interpretable than many nonlinear classifiers.

## Training Objective

Logistic regression is trained with log loss, also called binary cross-entropy:

$$
-y\log(p) - (1-y)\log(1-p)
$$

This rewards well-calibrated probabilities, not only correct classes.

## Thresholds

The model outputs probabilities. A threshold turns probabilities into classes.

The default threshold is often 0.5, but that is not always right.

Choose threshold based on:

- Precision-recall tradeoff.
- Cost of false positives.
- Cost of false negatives.
- Review capacity.
- Business constraints.

## Regularization

Use regularization to reduce overfitting:

- L2 for coefficient shrinkage.
- L1 for sparse feature selection.
- Elastic net for a mix.

Scale features when using regularized logistic regression.

## Evaluation

Useful metrics:

- Accuracy.
- Precision.
- Recall.
- F1.
- ROC-AUC.
- PR-AUC.
- Log loss.
- Calibration.

For imbalanced data, accuracy can be misleading. Precision-recall curves are often more useful.

## Closing

Logistic regression is a strong baseline for classification. It is fast, interpretable, and useful for understanding whether the features contain predictive signal.
