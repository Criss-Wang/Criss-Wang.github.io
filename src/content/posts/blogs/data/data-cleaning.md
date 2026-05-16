---
title: "The Data Mining Trilogy II: Cleaning"
date: 2019/09/01
categories:
  - Blogs
tags:
  - Data Mining
  - Data Cleaning
excerpt: "A practical guide to cleaning data: missing values, duplicates, outliers, inconsistent categories, schema checks, and reproducible cleaning pipelines."
layout: post
toc: true
---

## Introduction

Data cleaning turns raw data into data that is consistent enough to analyze or model. The goal is not to make data perfect. The goal is to make data trustworthy for the task.

Cleaning should be reproducible. If a notebook cell fixes a problem by hand, move that logic into a pipeline before the project becomes important.

## Missing Values

First ask why the value is missing.

Possible causes:

- The value does not apply.
- The value was not collected.
- The user skipped it.
- A pipeline failed.
- The source system changed.

Treatment options:

- Leave missingness explicit.
- Add a missingness indicator.
- Impute with mean, median, mode, or model-based estimates.
- Drop rows when the missingness is rare and unbiased.
- Drop columns when the feature is mostly unavailable.
- Fail the pipeline when missingness means upstream breakage.

Do not impute automatically without understanding the cause.

## Duplicates

Duplicates can come from retries, joins, scraping, batch replays, or logging bugs.

Check:

- Exact duplicate rows.
- Duplicate primary keys.
- Same entity repeated at different granularities.
- Near-duplicates in text or documents.

Deduplication rules should be explicit. Keep the most recent record, the first record, or an aggregated record only when the rule matches the task.

## Outliers

Outliers can be errors or real rare events.

Examples:

- Negative age is probably invalid.
- A large transaction may be legitimate.
- A very long document may be a parsing issue.

Use domain rules, plots, and source checks before deleting outliers. For modeling, consider robust scaling, clipping, winsorization, or models less sensitive to extreme values.

## Inconsistent Categories

Categorical values often drift:

- `USA`, `US`, and `United States`.
- Case differences.
- Extra spaces.
- Legacy codes.
- New categories after deployment.

Normalize categories with explicit maps. For production systems, define behavior for unknown categories.

## Schema Validation

Validate the data shape:

- Required columns.
- Data types.
- Allowed ranges.
- Allowed categories.
- Unique keys.
- Nullability.
- Date ordering.

Schema checks should run every time the pipeline runs.

## Cleaning Without Leakage

Fit cleaning transforms only on training data when they learn from distributions.

Examples:

- Imputation values.
- Scaling parameters.
- Rare category thresholds.
- Text vocabulary.

Then apply the fitted transform to validation, test, and production data.

## Cleaning Checklist

Before modeling:

- Missing values are understood.
- Duplicates are handled.
- Outliers are reviewed.
- Categories are normalized.
- Types are correct.
- Date and time fields are consistent.
- Schema checks are automated.
- Cleaning steps are versioned.

Clean data is not data with no problems. It is data whose problems are visible and controlled.
