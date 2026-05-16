---
title: "Model Validations and Performance Evaluators"
date: 2019/05/29
categories:
  - Blogs
tags:
  - Machine Learning
  - Evaluation
excerpt: "A practical guide to validation splits, cross-validation, classification and regression metrics, calibration, slice evaluation, and model comparison."
layout: post
mathjax: true
toc: true
---

## Introduction

Model validation estimates how well a model will perform on data it has not seen. It is one of the most important parts of machine learning because training metrics mostly tell you how well the model fit the training data.

Good validation answers:

- Does the model generalize?
- Which failure cases matter?
- Is the improvement real?
- Is the model good enough for the product constraint?

## Splitting Data

The split should match production reality.

### Random Split

Use random splits when examples are independent and identically distributed.

### Time-Based Split

Use time-based splits when the model predicts future behavior from past data.

### Group Split

Use group splits when records from the same user, account, patient, or entity could leak across train and test.

### Cross-Validation

Use cross-validation when data is small and you need a more stable estimate. Be careful with time-dependent data; ordinary k-fold cross-validation can leak future information.

## Classification Metrics

Common metrics:

- **Accuracy:** simple, but misleading under class imbalance.
- **Precision:** of predicted positives, how many are correct.
- **Recall:** of actual positives, how many are found.
- **F1:** harmonic mean of precision and recall.
- **ROC-AUC:** ranking quality across thresholds.
- **PR-AUC:** often better for imbalanced positive classes.
- **Log loss:** probability quality.

Choose based on error cost. A fraud model, medical model, and spam filter may all need different precision-recall tradeoffs.

## Regression Metrics

Common metrics:

- **MAE:** average absolute error, robust and interpretable.
- **MSE:** penalizes large errors more strongly.
- **RMSE:** square root of MSE, same unit as the target.
- **MAPE:** percentage error, but unstable near zero.
- **R-squared:** variance explained, useful but not enough alone.

Plot residuals. A single aggregate metric can hide systematic underprediction or overprediction.

## Calibration

Calibration asks whether predicted probabilities match observed frequencies.

If a model predicts 0.8 probability for many cases, roughly 80 percent of those cases should be positive.

Calibration matters when probabilities drive decisions:

- Risk scoring.
- Medical triage.
- Fraud review queues.
- Ranking with expected value.

Use reliability diagrams, Brier score, and calibration by slice.

## Slice Evaluation

Aggregate performance is not enough.

Evaluate by:

- User segment.
- Geography.
- Language.
- Device.
- Data source.
- Time period.
- Label source.
- Important edge cases.

A model can improve overall while hurting the group that matters most.

## Model Comparison

Compare models under the same conditions:

- Same train/validation/test split.
- Same preprocessing.
- Same metric implementation.
- Same random seeds when possible.
- Same inference constraints.

For noisy results, use repeated runs or confidence intervals. For product launches, combine offline validation with online testing when possible.

## Validation Checklist

Before trusting a model:

- Baseline exists.
- Split matches production.
- No leakage is detected.
- Primary metric is defined.
- Guardrail metrics are defined.
- Slice metrics are reviewed.
- Calibration is checked when probabilities matter.
- Error examples are inspected.
- Test set is held out until final review.

Validation is not a formality. It is the evidence behind model trust.
