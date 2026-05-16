---
title: "A Fundamental Course for Data Engineering"
date: 2020/05/01
categories:
  - Blogs
tags:
  - Data Engineering
  - Big Data
excerpt: "A practical introduction to data engineering fundamentals: ingestion, storage, batch and streaming processing, orchestration, data quality, governance, and serving."
layout: post
toc: true
---

## Introduction

Data engineering builds the systems that move data from source systems to useful destinations: analytics tables, machine learning datasets, dashboards, search indexes, and product features.

A good data pipeline is reliable, observable, versioned, and understandable. It should not require heroic debugging every time a source changes.

## Core Responsibilities

Data engineering usually includes:

- Ingesting data from source systems.
- Storing raw and processed data.
- Transforming data into usable tables or features.
- Scheduling and orchestrating jobs.
- Validating quality.
- Managing metadata and lineage.
- Serving data to analytics, ML, and applications.

## Ingestion

Ingestion moves data into the platform.

Patterns:

- Batch file loads.
- Database replication.
- Event streams.
- API extraction.
- Log collection.

Important questions:

- Is ingestion idempotent?
- Can late data arrive?
- How are duplicates handled?
- What happens when the source schema changes?
- Is raw data preserved?

## Storage

Common storage layers:

- **Raw layer:** source data with minimal changes.
- **Clean layer:** validated and standardized data.
- **Curated layer:** business-ready tables, features, or aggregates.

Storage choices include warehouses, data lakes, lakehouses, databases, object storage, and search systems. The right choice depends on query patterns, cost, latency, governance, and scale.

## Batch Processing

Batch processing handles data in scheduled jobs. It is useful for:

- Daily reporting.
- Training datasets.
- Feature generation.
- Backfills.
- Large joins and aggregations.

Batch jobs should be repeatable. If a job is rerun for a date, the output should be predictable.

## Streaming Processing

Streaming handles events as they arrive.

Use it for:

- Real-time monitoring.
- Event enrichment.
- Fraud or anomaly detection.
- Low-latency features.

Streaming systems need careful handling of ordering, duplicates, late events, and state.

## Orchestration

Orchestration tools schedule and connect pipeline steps.

They should answer:

- What ran?
- When did it run?
- What failed?
- What data did it produce?
- What depends on it?

Airflow, Dagster, Prefect, and cloud-native orchestrators all solve variants of this problem.

## Data Quality

Validate data at multiple points:

- Schema.
- Freshness.
- Row count.
- Null rate.
- Unique keys.
- Allowed values.
- Distribution shifts.

Bad data should fail loudly when it threatens downstream users.

## Governance and Lineage

As systems grow, teams need to know:

- Where data came from.
- Who owns it.
- What it means.
- Who can access it.
- Which downstream assets depend on it.
- How long it should be retained.

Governance is not paperwork when the data powers decisions or models. It is operational safety.

## Closing

Data engineering is the foundation for analytics and machine learning. The best pipelines are not the most complex ones; they are the ones that can be rerun, trusted, monitored, and explained.
