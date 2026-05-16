---
title: "Ensemble Models: Bagging Techniques"
date: 2021/02/01
categories:
  - Blogs
tags:
  - Supervised Learning
  - Ensembles
excerpt: "A practical explanation of bagging, bootstrap sampling, random forests, out-of-bag evaluation, variance reduction, and when bagging helps."
layout: post
mathjax: true
toc: true
---

## Introduction

Bagging, short for bootstrap aggregating, trains multiple models on different bootstrap samples of the training data and combines their predictions.

It is most useful for high-variance models. Decision trees are the classic example.

## Bootstrap Sampling

A bootstrap sample is created by sampling from the training set with replacement. Some examples appear multiple times, and some are left out.

Each model sees a slightly different dataset. If the base learner is sensitive to data changes, this creates diverse models.

## Aggregation

For regression, bagging usually averages predictions:

$$
\hat{y} = \frac{1}{M}\sum_{m=1}^{M}\hat{y}_m
$$

For classification, it can use majority vote or averaged probabilities.

The aggregation reduces variance because independent errors cancel out.

## Random Forest

Random forest adds feature randomness to bagging.

Each tree is trained with:

- A bootstrap sample of rows.
- A random subset of features considered at each split.

This decorrelates the trees. Less correlation means averaging helps more.

## Out-of-Bag Evaluation

Because each tree leaves out some training examples, those out-of-bag examples can be used to estimate performance.

Out-of-bag evaluation is convenient, but still use a proper validation or test set for final evaluation when the decision matters.

## When Bagging Helps

Use bagging when:

- The base model has high variance.
- Training data is not extremely small.
- Interpretability of one simple model is less important than performance.
- Inference cost is acceptable.

Bagging helps less when the base model is already stable or when errors are highly correlated across models.

## Practical Notes

Tune:

- Number of trees.
- Maximum depth.
- Minimum samples per leaf.
- Number of features considered per split.
- Class weights for imbalanced data.

Random forests are strong tabular baselines because they handle nonlinear relationships, feature interactions, and mixed feature types with relatively little preprocessing.

## Closing

Bagging is a variance-reduction strategy. It works best when many unstable but useful models can be averaged into a more reliable predictor.
