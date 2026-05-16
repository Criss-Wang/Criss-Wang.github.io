---
title: "Hyperparameter Tuning"
date: 2019/06/25
categories:
  - Blogs
tags:
  - Machine Learning
  - Optimization
excerpt: "A practical guide to hyperparameter tuning with search spaces, validation design, random search, Bayesian optimization, early stopping, and experiment tracking."
layout: post
mathjax: true
toc: true
---

## Introduction

Hyperparameters are choices set before or around training: learning rate, tree depth, regularization strength, batch size, number of layers, retrieval top-k, and many others.

Tuning is the process of searching for hyperparameters that improve validation performance without overfitting the validation process itself.

## Start With a Baseline

Before tuning, build a baseline:

- Simple model.
- Clean validation split.
- Fixed metric.
- Reproducible config.
- Logged result.

If the baseline is unstable, tuning will mostly amplify noise.

## Define the Search Space

A good search space matters more than a fancy optimizer.

Use ranges that reflect the scale of the parameter:

- Learning rate: log scale.
- Regularization: log scale.
- Number of trees: linear or bounded integer.
- Tree depth: small integer range.
- Dropout: bounded continuous range.
- Batch size: powers of two or hardware-friendly values.

Example:

```python
search_space = {
    "learning_rate": ("loguniform", 1e-5, 1e-2),
    "weight_decay": ("loguniform", 1e-6, 1e-1),
    "batch_size": [16, 32, 64],
    "warmup_ratio": ("uniform", 0.0, 0.1),
}
```

Avoid tuning parameters that do not matter yet. Start with the few choices most likely to move the metric.

## Validation Design

Tuning quality depends on validation quality.

Choose:

- Random split for independent examples.
- Time-based split for time-dependent tasks.
- Group split for users, accounts, patients, or entities.
- Cross-validation when data is small.

Do not tune on the test set. The test set is for final estimation, not iterative decision-making.

## Search Methods

### Grid Search

Grid search tries every combination from a fixed grid. It is simple but inefficient when many parameters are irrelevant.

Use grid search for small spaces or when you need a controlled comparison.

### Random Search

Random search samples from distributions. It is often stronger than grid search for the same budget because it explores more values for important parameters.

Use random search as the default baseline for tuning.

### Bayesian Optimization

Bayesian optimization models the relationship between hyperparameters and metric results, then chooses promising next trials.

Use it when:

- Each run is expensive.
- The metric is reasonably stable.
- The search space is not too large or chaotic.

### Early Stopping and Successive Halving

Early stopping stops weak trials before they consume full budget. Successive halving and Hyperband allocate more resources to promising trials.

Use these when training runs are expensive and partial learning curves are predictive.

## Track Every Trial

For each trial, log:

- Hyperparameters.
- Metric.
- Data version.
- Code version.
- Random seed.
- Runtime.
- Hardware.
- Failure reason, if any.

Without tracking, tuning becomes folklore.

## Avoid Common Mistakes

### Overfitting the Validation Set

If you run hundreds of trials and pick the best validation score, you may overfit the validation split. Use a final held-out test set or nested validation when the decision is important.

### Ignoring Cost

The best metric may not be the best model. Include latency, memory, and training cost in the decision.

### Searching Too Widely Too Soon

Huge search spaces waste budget. Use prior knowledge and early experiments to narrow the range.

### Comparing Noisy Runs

If randomness is large, repeat important configurations with different seeds.

## Practical Tuning Order

For neural networks:

1. Learning rate.
2. Batch size and gradient accumulation.
3. Weight decay.
4. Warmup and schedule.
5. Dropout or regularization.
6. Architecture size.

For tree-based models:

1. Number of trees.
2. Learning rate.
3. Max depth.
4. Minimum samples per leaf.
5. Subsampling.
6. Regularization.

## Closing

Hyperparameter tuning is controlled experimentation. Define a meaningful search space, validate correctly, track every trial, and choose the model that satisfies both metric and system constraints.
