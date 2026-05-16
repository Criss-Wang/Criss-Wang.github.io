---
title: "Testing in Machine Learning"
excerpt: "A practical checklist for testing data, models, ML systems, and CI/CD pipelines."
date: 2024/04/19
categories:
  - Blogs
tags:
  - Evaluation
  - MLOps
layout: post
mathjax: true
toc: true
---

Machine learning testing is broader than ordinary software testing. A model can pass unit tests and still fail because the data distribution changed, the evaluation metric is wrong, the serving path is overloaded, or the product behavior is biased.

In MLOps, a useful testing strategy needs to cover four layers:

- **Data:** Are the inputs reliable, representative, and separated correctly?
- **Model:** Does the model improve the metric that matters, and does it stay robust under realistic variation?
- **System:** Can the training and inference pipelines run reliably at production scale?
- **Product:** Does the model behave safely and fairly for the people affected by it?

The sections below are a practical checklist for those layers.

## Data Quality and Diversity

Data quality is the first testing surface. If the data is inconsistent, leaked, mis-split, or unrepresentative, later model evaluation becomes difficult to trust.

### Data Consistency

Consistency checks validate the integrity, accuracy, completeness, shape, and range of the datasets used for training, validation, and testing. Common tools include data profiling, schema validation, anomaly detection, and simple invariant checks.

At minimum, test for:

- **ETL implementation errors:** Check parsing, joins, deduplication, missing-value handling, label generation, and error handling. Encoding issues are especially easy to miss and can quietly damage text-heavy ML systems.
- **Input/output shape and range mismatches:** Confirm that feature tensors, labels, prediction outputs, and post-processed values match the contracts expected by downstream code.
- **Train/validation/test split issues:** Look for class imbalance across splits, duplicated examples, time-travel leakage, user-level leakage, and contamination from training data into validation or test data.
- **Unexpected feature correlations:** Investigate correlations that look too strong to be real. They may reveal leakage, data collection artifacts, or temporal dependencies.

### Data Drift

Data drift is a change in the input distribution over time. Concept drift is a change in the relationship between inputs and labels. Both can degrade production performance even when the model code has not changed.

At minimum, monitor:

- **Distribution drift:** Use tests such as the Kolmogorov-Smirnov test for numerical features and the chi-square test for categorical features, then pair those tests with domain review so statistical noise does not trigger false alarms.
- **Performance drift:** Track production metrics when labels are available, and use proxy metrics or delayed-label analysis when labels arrive slowly.
- **Segment-level drift:** Watch important user groups, traffic sources, regions, devices, or product surfaces separately. Aggregate metrics can hide a regression in a smaller segment.

## Model Quality

Model quality testing asks whether the model is correct enough, stable enough, and useful enough for the product context. It should happen before full system testing, but it depends on a reliable evaluation framework.

### Evaluation Framework

Before regression or robustness testing, test the evaluation pipeline itself. It is easy to waste days improving a model against a broken metric.

Check that:

- The metric implementation matches the product objective.
- Label normalization, filtering, and aggregation rules are correct.
- Custom evaluators are tested with small examples where the expected answer is obvious.
- Offline metrics are compared against a current baseline, not just against an isolated candidate score.

### Regression Testing

Regression testing checks whether a new model, prompt, feature, or serving change makes existing behavior worse. Regression can happen during training or inference, so it should be covered in development, staging, and production.

At minimum, test:

- **Training behavior:** convergence, overfitting, underfitting, and sensitivity to random seeds.
- **Inference behavior:** prediction contracts, batch and streaming paths, timeout behavior, and response formatting.
- **Directional expectations:** examples where the expected direction is obvious, such as a snow prediction becoming less likely when temperature rises.
- **Product acceptance cases:** user-facing workflows that must keep working even when the model changes.

### Robustness

Robustness testing checks whether the model behaves consistently under perturbations, adversarial inputs, noisy data, and out-of-distribution examples.

Useful methods include:

- Input perturbation tests, such as typos, paraphrases, feature noise, cropping, or missing fields.
- Adversarial or stress examples targeted at known model weaknesses.
- Robust training or optimization methods when the risk justifies the extra complexity.
- Evaluation on slices that represent edge cases, minority classes, or important production segments.

## Fairness and Bias

Fairness testing is part of product quality, not an optional ethics appendix. The relevant risks depend heavily on the domain, but teams should explicitly test for disparate performance, representational gaps, and harmful outputs.

For traditional ML systems, this can mean testing metrics across demographic groups, auditing labels and features for proxy variables, and monitoring bias after launch. For LLM systems, it can also mean alignment evaluations, red-team prompts, harmful-output checks, and reviewer audits.

There is no single universal baseline here. The right tests depend on the affected users, business context, legal constraints, and product failure modes. A good starting point is [IBM AI Fairness 360](https://aif360.res.ibm.com/), which catalogs fairness metrics and mitigation techniques. LLM-based evaluators can help with review at scale, but they also introduce their own evaluation bias, so they should not be treated as the only judge.

## System Testing

System testing checks whether the training and inference pipelines can run reliably, recover from failure, and meet production constraints. The details differ between training and serving, but both need correctness, scalability, and fault tolerance.

For training systems, test:

- Multi-node or multi-server recovery through checkpointing, retry logic, and replica synchronization.
- CPU, GPU, memory, and storage utilization.
- Dataset loading throughput and storage bottlenecks.
- Network congestion, latency, and distributed-training communication overhead.

For inference systems, test:

- Batch, streaming, and real-time serving paths.
- Latency under normal load, peak load, and stress conditions.
- Error handling for provider failures, malformed inputs, oversized requests, and timeout paths.
- Changes in processing time when servers are under sustained load.

## ML Testing and Software Engineering

High-quality ML code still needs ordinary software engineering discipline. The difference is that ML tests often need to avoid large fixtures, expensive models, and hidden assumptions about learned weights.

Good unit-test practices include:

- **Use tiny examples:** Prefer one or two inline examples over loading a full data file.
- **Test random or empty weights when possible:** This checks architecture and tensor-flow assumptions without depending on a specific trained model.
- **Keep critical model tests separate:** Mark slow tests clearly and run them before important merges or releases.
- **Test post-processing logic:** Recommendation filtering, diversification, ranking rules, thresholds, and formatting often contain product-critical logic outside the model itself.

## ML Testing in CI/CD

ML testing should be continuous, but it is not always deterministic. Data changes, stochastic training, and human evaluation can make CI/CD harder than in traditional software projects.

A practical pipeline usually combines:

- Fast unit tests on every commit.
- Data validation and metric checks on model-training jobs.
- Regression suites for candidate models, prompts, and feature changes.
- Slower integration or end-to-end tests before deployment.
- Human review for ambiguous product, fairness, or quality judgments.
- Production monitoring for drift, latency, cost, and model-output quality.

Jeremy Jordan's testing guide has a helpful diagram of the model-development loop. The key point is the same one used throughout this post: combine software tests, data tests, model behavior tests, and production monitoring instead of treating validation score as the only check.

For a deeper enterprise MLOps view, the [Databricks Big Book of MLOps](https://www.databricks.com/resources/ebook/the-big-book-of-mlops) is a useful reference for how development, UAT, production testing, and monitoring fit together.

## References

- [Jeremy Jordan: Effective testing for machine learning systems](https://www.jeremyjordan.me/testing-ml/)
- [Eugene Yan: How to Test Machine Learning Code and Systems](https://eugeneyan.com/writing/testing-ml/)
- [Tekhnoal: Load tests for ML models](https://www.tekhnoal.com/load-tests-for-ml-models)
