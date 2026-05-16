---
title: "The Data Mining Trilogy III: Analysis"
date: 2020/09/03
categories:
  - Blogs
tags:
  - Data Mining
  - EDA
excerpt: "A practical guide to exploratory data analysis: distributions, relationships, missingness, outliers, leakage checks, segmentation, and communicating findings."
layout: post
toc: true
---

## Introduction

Exploratory data analysis (EDA) is the process of understanding data before making strong claims or training models.

Good EDA answers:

- What does each variable look like?
- What data quality issues exist?
- How do variables relate to the target?
- Which groups behave differently?
- What assumptions might fail?
- What should be tested next?

EDA is not a gallery of plots. It is structured curiosity.

## Start With Shape and Schema

Check:

- Number of rows and columns.
- Column types.
- Missing values.
- Duplicate keys.
- Time range.
- Target distribution.
- Granularity.

These basics often reveal the biggest issues.

## Study Distributions

For numeric variables:

- Histogram.
- Quantiles.
- Mean and median.
- Standard deviation.
- Minimum and maximum.
- Outlier counts.

For categorical variables:

- Unique values.
- Top categories.
- Rare categories.
- Missing category behavior.

Look for impossible values, heavy tails, and categories that should be merged.

## Analyze Relationships

Useful checks:

- Correlation among numeric features.
- Feature distribution by target.
- Target rate by category.
- Pair plots for small feature sets.
- Time trends.
- Segment-level behavior.

Do not overinterpret correlation. EDA suggests hypotheses; it does not prove causality.

## Check Leakage

EDA is a good time to catch leakage.

Warning signs:

- A feature is almost perfectly correlated with the target.
- A timestamp occurs after the prediction time.
- Aggregates include future data.
- Train and test sets contain duplicate entities.
- Text contains labels or answer keys.

If a feature looks too good, investigate.

## Segment the Data

Aggregate plots can hide important differences.

Segment by:

- Time period.
- Geography.
- Device.
- Customer type.
- Language.
- Product.
- Data source.
- Label source.

These slices often become validation slices later.

## Communicate Findings

A useful EDA report should include:

- What was analyzed.
- Key data quality issues.
- Important distributions.
- Surprising relationships.
- Leakage risks.
- Modeling implications.
- Recommended next steps.

Keep the report short enough for a teammate to act on it.

## Closing

EDA is where you build judgment about the dataset. The output should not just be plots; it should be decisions about cleaning, validation, feature engineering, and modeling.
