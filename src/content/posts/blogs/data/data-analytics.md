---
title: "An Overview of Big Data Analytics"
date: 2020/01/02
categories:
  - Blogs
tags:
  - Data
  - Analytics
excerpt: "A concise overview of big data analytics: batch and streaming workloads, descriptive and predictive analysis, data quality, storage, compute, and communication."
layout: post
toc: true
---

## Introduction

Big data analytics is the practice of extracting useful information from data that is too large, fast, varied, or distributed for simple manual analysis.

The point is not that the data is "big." The point is that ordinary tools and workflows are no longer enough.

## Types of Analytics

### Descriptive Analytics

Descriptive analytics explains what happened.

Examples:

- Revenue dashboards.
- User activity summaries.
- Operational reporting.
- Funnel analysis.

### Diagnostic Analytics

Diagnostic analytics explains why something happened.

Examples:

- Investigating a conversion drop.
- Finding the source of increased latency.
- Comparing behavior across user segments.

### Predictive Analytics

Predictive analytics estimates what may happen next.

Examples:

- Churn prediction.
- Demand forecasting.
- Fraud scoring.
- Recommendation ranking.

### Prescriptive Analytics

Prescriptive analytics recommends actions.

Examples:

- Inventory planning.
- Pricing decisions.
- Next-best-action systems.

## Batch and Streaming

Batch analytics processes data in scheduled chunks. It is good for reports, backfills, training datasets, and daily metrics.

Streaming analytics processes events as they arrive. It is useful for monitoring, anomaly detection, event enrichment, and low-latency systems.

Many real systems use both: streaming for freshness and batch for correctness, reconciliation, or heavy processing.

## Data Quality

Analytics quality depends on data quality.

Track:

- Missing values.
- Duplicate events.
- Schema changes.
- Late-arriving data.
- Invalid timestamps.
- Bot or spam traffic.
- Inconsistent identifiers.

Dashboards can look polished while being wrong. Build data quality checks into the pipeline.

## Storage and Compute

Common storage patterns:

- Data warehouse for structured analytics.
- Data lake for raw and semi-structured data.
- Lakehouse for combining storage flexibility with table management.
- Object storage for large files.
- Search index for text and log exploration.

Common compute patterns:

- SQL engines for analysis.
- Spark for distributed batch processing.
- Stream processors for event pipelines.
- Python or R for exploration and modeling.

Choose based on access patterns, not tool popularity.

## Communicating Results

Analytics work is only useful when it changes understanding or action.

A good analysis states:

- Question.
- Data source.
- Time range.
- Method.
- Key findings.
- Limitations.
- Recommended action.

Avoid burying the conclusion under charts. Say what changed, why it matters, and how confident you are.

## Closing

Big data analytics is an engineering and reasoning discipline. Reliable pipelines, clear metrics, data quality checks, and careful communication matter as much as the computation engine.
