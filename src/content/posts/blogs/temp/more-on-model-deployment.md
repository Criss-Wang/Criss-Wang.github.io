---
title: "More on Model Deployment"
date: 2021-11-01
categories:
  - Blogs
  - Draft Notes
tags:
  - MLOps
  - Deployment
excerpt: "A practical overview of model deployment patterns, artifact promotion, online and batch serving, rollout strategies, rollback, and production monitoring."
layout: post
toc: true
---

## Introduction

Model deployment is the step where a trained model becomes part of a product or workflow. It is not only "put the model behind an API." A deployed model needs an artifact, runtime, interface, rollout plan, rollback path, logs, and monitoring.

This post is a compact deployment checklist.

## What Gets Deployed

Deploy more than weights.

A deployable model package should include:

- Model artifact.
- Loading code.
- Preprocessing code.
- Postprocessing code.
- Input and output schema.
- Tokenizer or feature definitions.
- Runtime dependencies.
- Model version.
- Evaluation report.
- Owner.

If any of these are missing, the deployment is fragile.

## Serving Patterns

### Online Serving

Online serving is request-response inference. Use it when users or downstream services need predictions immediately.

Examples:

- Fraud scoring.
- Search ranking.
- Real-time recommendations.
- LLM assistant responses.

Watch latency, availability, error rate, cost per request, and fallback behavior.

### Batch Serving

Batch serving runs predictions over many records on a schedule.

Examples:

- Daily customer scores.
- Weekly demand forecasts.
- Offline document embeddings.
- Recommendation candidate generation.

Batch serving needs monitoring too. A failed batch job can silently poison downstream dashboards or products.

### Streaming Serving

Streaming serving processes events as they arrive. It is useful when the system reacts to event streams but does not need a direct user response for each event.

Examples:

- Real-time anomaly detection.
- Event enrichment.
- Monitoring pipelines.

## Rollout Strategies

Choose rollout strategy based on risk.

- **Direct rollout:** simple replacement, useful for low-risk internal systems.
- **Shadow deployment:** run the new model without affecting users.
- **Canary deployment:** send a small percentage of traffic to the new model.
- **A/B test:** compare variants against product metrics.
- **Champion/challenger:** keep a current production model while testing alternatives.

Shadow and canary deployments are especially useful when offline metrics do not fully predict production behavior.

## Rollback

Rollback should be designed before launch.

You need:

- Previous model artifact.
- Previous config.
- Traffic routing control.
- Database or feature compatibility.
- A decision rule for rollback.
- A human owner who can execute it.

Rollback fails when the new model changes schemas or downstream assumptions without compatibility planning.

## Monitoring

Monitor both service health and model health.

Service metrics:

- Request rate.
- Latency distribution.
- Error rate.
- Timeout rate.
- Resource usage.
- Queue depth.
- Cost.

Model metrics:

- Prediction distribution.
- Confidence distribution.
- Input feature distribution.
- Drift.
- Slice-level performance when labels arrive.
- Human feedback.
- Safety or policy violations, if relevant.

Monitoring should include model version. Otherwise you cannot connect a production issue to the artifact that caused it.

## Security and Privacy

Deployment often exposes data paths that training did not.

Check:

- Authentication.
- Authorization.
- Secret management.
- Input validation.
- Rate limits.
- PII handling.
- Log redaction.
- Dependency vulnerabilities.

For LLM systems, also consider prompt injection, data exfiltration, tool-call permissions, and retrieval source trust.

## Closing

Good deployment makes model behavior observable and reversible. A model that cannot be rolled back, monitored, or explained operationally is not production-ready, even if its validation score is strong.

For a broader post-training view, see [MLOps Post-Training Considerations](/writing/blogs/mlops/ml-post-training/).
