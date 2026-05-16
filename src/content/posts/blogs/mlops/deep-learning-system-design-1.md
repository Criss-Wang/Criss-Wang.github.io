---
title: "Deep Learning System Design: A Checklist, Part I"
excerpt: "A practical checklist for the early stages of a deep learning system: data, modeling, evaluation, training, and experiment tracking."
date: 2024/02/09
categories:
  - Blogs
tags:
  - Deep Learning
  - System Design
layout: post
mathjax: true
toc: true
---

## Introduction

A deep learning system is more than a trained model. The model sits inside a system of data pipelines, evaluation rules, training infrastructure, experiment records, deployment paths, serving contracts, and monitoring loops.

This first part focuses on the pre-deployment side:

1. Data.
2. Modeling.
3. Evaluation.
4. Training and optimization.
5. Experiments.

Part II covers packaging, serving, deployment, and monitoring.

## Step 1: Data

Data quality decides how much intelligence the model can learn. Before modeling, answer the boring questions carefully.

### Data Source

Check:

- What data exists?
- Who owns it?
- How is access granted?
- How fresh is it?
- How large is it?
- Which user groups, regions, languages, or edge cases are missing?
- Does it include sensitive or regulated information?
- Is there user feedback or operational data that can improve the system later?

Logs are useful, but they should not be kept forever by default. Keep them long enough for debugging, audit, and monitoring needs, then apply retention rules.

### Data Format and Storage

The storage format should match the workload.

- Row-oriented formats are often convenient for transactional writes.
- Columnar formats such as Parquet are often better for analytical reads.
- Object storage is common for raw artifacts, training snapshots, and large files.
- Databases are useful for structured operational records and metadata.

Separate app data from ML artifacts when the access patterns differ. A product database, feature store, model registry, and experiment tracker usually have different responsibilities.

### ETL and Feature Pipelines

A data pipeline should make transformations repeatable:

- Extract raw data from trusted sources.
- Validate schema and basic quality.
- Transform into model-ready features, examples, chunks, or embeddings.
- Write immutable or versioned outputs.
- Record lineage from raw source to training set.

The question is not only "Can I make a training CSV?" It is "Can someone rebuild the same training data next month?"

### Data Quality

Validate:

- Missing values.
- Duplicate records.
- Invalid ranges.
- Broken categorical values.
- Unexpected schema changes.
- Label distribution changes.
- Train/test leakage.
- Personally identifiable information.
- Outliers that are real but rare.
- Outliers caused by data errors.

For tabular pipelines, libraries such as Pandera or Great Expectations can make these checks explicit. For unstructured data, the same principle applies: validate document metadata, file integrity, language, length, source, and parsing success.

## Step 2: Modeling

Start with the task, not with the model.

### Task Definition

Clarify:

- Is the output a class, score, ranking, text, image, embedding, action, or forecast?
- Are labels available?
- Are labels reliable?
- Does the model need to explain its output?
- Does the output trigger an automatic action or human review?
- What latency and cost are acceptable?

These answers narrow the model family before architecture debates begin.

### Baselines

Every project needs a baseline:

- Random baseline.
- Majority-class or simple heuristic baseline.
- Existing production workflow.
- Classical ML model.
- Small neural network.
- Retrieval-only or rules-only baseline for LLM systems.

Baselines protect the team from celebrating a complex model that barely beats a simple solution.

### Metric Selection

Pick metrics that match the failure cost.

- Classification: precision, recall, F1, ROC-AUC, PR-AUC, calibration, confusion matrix.
- Regression: MAE, RMSE, MAPE, pinball loss, error by segment.
- Ranking and retrieval: recall@k, precision@k, MRR, nDCG.
- Generation: task success, factuality, citation quality, refusal behavior, style, latency, cost.
- Segmentation: IoU, Dice, pixel accuracy, per-class scores.
- Forecasting: error by horizon, seasonality, and segment.
- Reinforcement learning: average return, success rate, safety constraints, sample efficiency.

Do not stop at one metric. Add slice-level evaluation for important user groups, data sources, and edge cases.

### Model Comparison

When comparing models, make the comparison fair:

- Same train/validation/test split.
- Same data version.
- Same evaluation code.
- Same inference constraints.
- Confidence intervals or repeated runs when randomness matters.
- Human review for qualitative tasks.

Useful checks include perturbation tests, invariance tests, directional expectation tests, calibration checks, and slice-based evaluation.

## Step 3: Training and Optimization

Training quality depends on stability, memory, throughput, and observability.

Before scaling up:

- Overfit a tiny batch to catch implementation bugs.
- Confirm the loss decreases.
- Confirm labels and predictions are aligned.
- Set seeds where reproducibility matters.
- Log gradient norms and learning rate.
- Validate checkpoint resume.

Common training choices:

- Optimizer: SGD, Adam, AdamW, Adafactor, or task-specific variants.
- Scheduler: warmup plus cosine decay, linear decay, step decay, or reduce-on-plateau.
- Precision: FP32, FP16, BF16, or mixed precision.
- Memory strategy: smaller micro-batch, gradient accumulation, activation checkpointing, sharding, or offload.
- Scale strategy: single GPU, DDP, FSDP, tensor parallelism, or pipeline parallelism.

For deeper training notes, see [Deep Learning Training: A Practical Guide](/writing/blogs/mlops/ml-training/).

## Step 4: Experiments

Experiment tracking is not optional once runs become expensive, collaborative, or user-facing.

Track:

- Code version.
- Config and command-line arguments.
- Data version.
- Environment or container image.
- Hyperparameters.
- Metrics.
- Slice metrics.
- Evaluation artifacts.
- Checkpoints.
- Logs and errors.
- Hardware and runtime.

Tools such as MLflow, Weights & Biases, Neptune, and cloud-native trackers can help, but the tool is less important than the record. A run should answer:

- What did we try?
- What data did it use?
- What code produced it?
- What changed from the previous run?
- Did it improve the metric that matters?
- Is it reproducible?

### Task-Specific Artifacts

Different tasks need different artifacts:

- Classification: confusion matrix, ROC/PR curves, misclassified examples.
- Computer vision: predictions, masks, bounding boxes, overlays.
- NLP and LLM: prompts, retrieved context, generated outputs, refusal cases, citation checks.
- Ranking: top-k examples, failed queries, query slices.
- Tabular ML: feature importance, calibration plots, partial dependence, SHAP summaries when useful.
- Reinforcement learning: episode return, episode length, environment seed, videos or trajectories.

The goal is to make debugging possible without rerunning every experiment.

## Closing

The early system-design work is about discipline: reliable data, honest baselines, meaningful evaluation, stable training, and complete experiment records.

If these pieces are weak, deployment will only make the weakness more expensive. Once they are solid, move to packaging, serving, and monitoring in [Part II](/writing/blogs/mlops/deep-learning-system-design-2/).
