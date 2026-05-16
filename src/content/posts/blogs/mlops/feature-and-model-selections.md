---
title: "Feature Selection and Model Selection"
date: 2019/06/04
categories:
  - Blogs
tags:
  - Machine Learning
  - Model Selection
excerpt: "A practical guide to selecting features, choosing models, avoiding leakage, comparing validation results, and balancing accuracy with complexity."
layout: post
mathjax: true
toc: true
---

## Introduction

Feature selection and model selection are both attempts to answer the same question:

```text
Which representation and model should we trust for this task?
```

Feature selection chooses the inputs. Model selection chooses the learning algorithm and configuration. Both affect accuracy, interpretability, cost, and reliability.

## Feature Selection Goals

Feature selection can help:

- Reduce overfitting.
- Improve interpretability.
- Reduce training and inference cost.
- Remove noisy or redundant inputs.
- Make debugging easier.
- Reduce data collection burden.

But removing features can also remove signal. Always compare against a baseline with clear validation.

## Feature Selection Methods

### Filter Methods

Filter methods score features without training the final model.

Examples:

- Variance threshold.
- Correlation filtering.
- Mutual information.
- Chi-square tests for categorical features.

They are fast and useful for cleanup, but they may miss interactions between features.

### Wrapper Methods

Wrapper methods evaluate feature subsets by training models.

Examples:

- Forward selection.
- Backward elimination.
- Recursive feature elimination.

They can work well but are computationally expensive and can overfit validation data if repeated too many times.

### Embedded Methods

Embedded methods select features as part of model training.

Examples:

- L1-regularized linear models.
- Tree-based feature importance.
- Gradient-boosted model importance.

They are practical, but importance scores can be biased. Correlated features and high-cardinality categorical variables need extra care.

## Avoid Leakage

Feature selection must happen inside the training pipeline, not before the train/test split.

Bad pattern:

```text
Use all data to select features, then split into train and test.
```

Better pattern:

```text
Split data, fit feature selection on training folds, evaluate on held-out folds.
```

Leakage often creates beautiful validation numbers and painful production failures.

## Model Selection

Start with simple baselines:

- Majority-class baseline.
- Linear or logistic regression.
- Tree-based baseline.
- Existing business rule.

Then compare more complex models:

- Random forest.
- Gradient boosting.
- Neural networks.
- Retrieval or embedding-based methods.
- Domain-specific architectures.

Choose based on the full system, not only validation score.

## Comparison Criteria

Compare models by:

- Primary metric.
- Slice metrics.
- Calibration.
- Latency.
- Training cost.
- Inference cost.
- Interpretability.
- Data requirements.
- Operational complexity.
- Failure modes.

A model with slightly lower accuracy may be the better engineering choice if it is faster, easier to explain, and more reliable.

## Validation Strategy

Use validation that matches the real task:

- Random split for independent examples.
- Time-based split for forecasting or time-dependent behavior.
- Group split when records from the same user, account, or entity could leak.
- Cross-validation when data is small.
- Nested cross-validation when heavy model selection is part of the process.

The validation method is part of the model selection decision.

## Closing

Feature selection and model selection are not one-time rituals. They are controlled experiments. Start simple, avoid leakage, compare fairly, and choose the model that best satisfies the product and engineering constraints.
