---
title: "Testing Machine Learning Systems"
date: 2024-02-17
categories:
  - Blogs
  - Draft Notes
tags:
  - MLOps
  - Testing
excerpt: "A compact guide to unit tests, data tests, model behavior tests, evaluation, regression tests, and production checks for machine learning systems."
layout: post
toc: true
---

## Introduction

Machine learning systems need both software tests and model tests.

Software tests ask whether the code behaves as expected. Model tests ask whether the learned behavior is acceptable. You need both because a pipeline can be perfectly implemented and still produce a bad model.

## Software Tests

A normal software test suite still matters:

- **Unit tests:** fast tests for small functions.
- **Integration tests:** tests across modules, services, or pipeline stages.
- **Regression tests:** tests that reproduce previously fixed bugs.
- **Contract tests:** tests for schemas, API behavior, and interface expectations.

In ML systems, contract tests are especially valuable because training, serving, and monitoring often depend on the same feature and schema assumptions.

## Data Tests

Data tests catch problems before training or inference.

Check:

- Required columns.
- Data types.
- Missing-value ranges.
- Duplicate keys.
- Invalid categories.
- Label distribution.
- Feature ranges.
- Train/test leakage.
- Time ordering.
- Privacy constraints.

When possible, fail early. A broken data pipeline should not quietly produce a trained model.

## Pre-Training Model Tests

Before an expensive run:

- Overfit a tiny batch.
- Confirm loss decreases.
- Confirm output shape.
- Confirm labels align with examples.
- Check gradients are finite.
- Run one evaluation pass.
- Save and load a checkpoint.

These tests are cheap and catch many expensive bugs.

## Post-Training Behavior Tests

After training, evaluate expected behavior explicitly.

Useful test types:

- **Invariance tests:** perturb inputs in ways that should not change the output.
- **Directional expectation tests:** change an input in a way that should move prediction in a known direction.
- **Slice tests:** evaluate important subgroups or edge cases.
- **Counterfactual tests:** compare similar examples that differ in one important feature.
- **Regression examples:** keep cases that previously failed.
- **Calibration tests:** check whether predicted probabilities mean what they claim.

These tests make model quality more concrete than one aggregate metric.

## Evaluation and Tests Work Together

Evaluation summarizes model performance. Tests enforce specific expectations.

For example:

- Evaluation says F1 improved from 0.81 to 0.84.
- A model test says performance on a high-risk slice must not drop below 0.75 recall.
- A contract test says the serving schema must include all required features.

The model should pass all three before deployment.

## Production Tests

After deployment, keep checking:

- Latency.
- Error rate.
- Prediction distribution.
- Input drift.
- Data quality.
- Feedback quality.
- Label-based performance when labels arrive.
- Model version and rollback path.

Production checks are not optional. Models decay when the world changes.

## Closing

Testing ML systems is about making assumptions executable. If the team believes a behavior must hold, write a test for it. If a production failure happens, turn it into a regression test.

For a fuller version of this topic, see [Testing in Machine Learning](/writing/blogs/mlops/ml-testing/).

## Reference

- [Jeremy Jordan: Effective Testing for Machine Learning Systems](https://www.jeremyjordan.me/testing-ml/)
