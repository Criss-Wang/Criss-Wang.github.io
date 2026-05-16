---
title: "Deep Learning System Design: A Checklist, Part II"
excerpt: "A practical checklist for the production side of deep learning systems: packaging, deployment, serving, monitoring, logging, and model operations."
date: 2024/02/10
categories:
  - Blogs
tags:
  - Deep Learning
  - System Design
layout: post
mathjax: true
toc: true
---

## Recap

[Part I](/writing/blogs/mlops/deep-learning-system-design-1/) covered the early system-design work: data, modeling, evaluation, training, and experiment tracking.

This part covers the production side:

1. Packaging and model artifacts.
2. Deployment.
3. Serving.
4. Monitoring.
5. Logging and operations.

This is where a trained model becomes a system that users and other services can depend on.

## Step 5: Packaging and Model Artifacts

Packaging means turning a trained model into a reproducible artifact that another process can load safely.

A production artifact should include:

- Model weights.
- Model architecture or loading code.
- Preprocessing and postprocessing code.
- Input and output schema.
- Tokenizer, vocabulary, label map, or feature definitions.
- Dependency versions.
- Training config.
- Evaluation report.
- Model version.
- Data version.
- Owner and approval status.

The artifact should answer a simple question: "Can this model be loaded and evaluated without guessing what produced it?"

### Storage Choices

Common storage patterns:

- Object storage for raw artifacts.
- Model registry for version, metadata, stage, and approval status.
- Container image for runtime dependencies.
- Package repository for shared model code.
- Feature store or metadata store for feature contracts.

Avoid mystery files. A file named `best_model_final_v7.pt` is not a deployment strategy.

## Step 6: Deployment

Deployment is the process of promoting a model into an environment where it can be used.

Common patterns:

- **Direct replacement:** swap the old model for the new one.
- **Shadow deployment:** run the new model beside production without affecting users.
- **Canary deployment:** send a small amount of traffic to the new model.
- **A/B test:** split traffic between variants and compare outcomes.
- **Champion/challenger:** keep a production champion while evaluating challengers.
- **Batch deployment:** run scheduled jobs and write predictions to storage.

The right pattern depends on risk. A low-stakes internal classifier may only need a simple rollout. A high-traffic ranking model should usually use shadowing, canaries, and rollback.

### Deployment Checklist

Before rollout:

- The artifact loads in the target environment.
- Input schema validation is in place.
- The model passes offline evaluation gates.
- Latency and memory are measured.
- Fallback behavior exists.
- Rollback is tested.
- Logs and metrics are connected.
- The owner is clear.

## Step 7: Serving

Serving turns the model into a callable capability.

Decide:

- Online, batch, streaming, or offline inference.
- REST, gRPC, WebSocket, queue, or scheduled job interface.
- CPU, GPU, or specialized accelerator.
- Single model server or separate application and model services.
- Synchronous or asynchronous response.
- Maximum request size.
- Timeout and retry behavior.
- Input validation and output validation.
- Authentication and authorization.

### Online Serving

Online serving is request-response inference. It is appropriate when users or downstream services need fresh predictions immediately.

Watch:

- p50, p95, and p99 latency.
- Error rate.
- Timeout rate.
- Cold starts.
- GPU utilization.
- Batch size and queue time.
- Dependency failures.

### Batch Serving

Batch serving is useful when predictions can be computed on a schedule.

Good use cases:

- Daily risk scores.
- Offline recommendations.
- Document enrichment.
- Embedding refresh.
- Backfills.

Batch jobs still need monitoring. Silent failure can be worse than an obvious API error.

### Feedback Collection

Serving should preserve enough information for debugging and improvement:

- Request ID.
- Model version.
- Input metadata.
- Prediction.
- Confidence or score.
- Latency.
- User or system feedback when available.
- Ground truth label when it arrives.

Be careful with privacy. Log enough to debug, not everything by default.

## Step 8: Monitoring

Monitoring should cover both the system and the model.

### System Metrics

Track:

- Throughput.
- Latency distribution.
- Availability.
- Error rate.
- Timeout rate.
- CPU, GPU, memory, disk, and network usage.
- Queue depth.
- Dependency health.
- Cost.

These metrics tell you whether the service is operationally healthy.

### Model Metrics

Track:

- Prediction distribution.
- Confidence distribution.
- Feature distribution.
- Data drift.
- Concept drift when labels arrive.
- Slice-level performance.
- Calibration.
- Outlier rate.
- Human review outcomes.
- Feedback quality.

These metrics tell you whether the model is still behaving well.

### Logging

Use structured logs with timestamps, severity, request IDs, model versions, and relevant metadata.

Good logs make it possible to answer:

- Which model produced this prediction?
- Which input schema version did it use?
- Which dependency failed?
- Was latency caused by preprocessing, model inference, or downstream calls?
- Did this issue affect one request, one customer, or the whole system?

## Step 9: Updating the Model

Model updates should be deliberate.

Trigger retraining or replacement when:

- Data drift is persistent.
- Performance drops on important slices.
- New labels show behavior has changed.
- Product requirements change.
- New data sources become available.
- A security or safety issue appears.

Do not retrain blindly on a schedule if nobody reviews whether the new model is better. Automated retraining still needs gates.

## Production Quality Bar

A production ML system should be:

- **Scalable:** handles expected workload with margin.
- **Maintainable:** code, data, and artifacts are understandable.
- **Adaptable:** supports updates without rebuilding everything.
- **Reliable:** fails safely and recovers predictably.
- **Traceable:** decisions can be connected to data, code, and model version.

The model is only one part of that quality bar.

## Closing

Deep learning systems become valuable when the model is surrounded by engineering discipline: packaged artifacts, tested deployment paths, clear serving contracts, monitoring, logs, and ownership.

The checklist is not meant to slow the team down. It is meant to keep the team from discovering production requirements after users already depend on the system.

## Reference

- [Jeremy Jordan: Effective Testing for Machine Learning Systems](https://www.jeremyjordan.me/testing-ml/)
- [Jeremy Jordan: ML Monitoring](https://www.jeremyjordan.me/ml-monitoring/)
