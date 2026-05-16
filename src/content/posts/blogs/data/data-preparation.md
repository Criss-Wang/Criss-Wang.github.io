---
title: "The Data Mining Trilogy I: Preparation"
date: 2019/08/25
categories:
  - Blogs
tags:
  - Data Mining
  - Data Preparation
excerpt: "A practical overview of data preparation: defining the problem, collecting data, building schemas, splitting datasets, and preventing leakage."
layout: post
toc: true
---

## Introduction

Data preparation is the work before modeling: define the problem, collect the data, understand its structure, and create datasets that can support valid analysis or machine learning.

Poor preparation creates downstream confusion. Good preparation makes the rest of the project measurable.

## Define the Question

Start with the question:

- What decision will this data support?
- What is the target variable?
- What is the prediction time?
- What information is available at that time?
- What population does the dataset represent?
- What errors are costly?

For machine learning, the prediction time is critical. Any feature created after that time can leak future information.

## Collect the Data

Record:

- Source.
- Owner.
- Refresh frequency.
- Granularity.
- Time range.
- Known missing fields.
- Access restrictions.
- Privacy constraints.

Do not treat data collection as a one-time download. Data sources change.

## Understand Granularity

Granularity defines what one row means.

Examples:

- One row per user.
- One row per transaction.
- One row per session.
- One row per product per day.
- One row per document chunk.

Many bugs come from mixing granularities without noticing. For example, joining user-level features to transaction-level labels can duplicate values and distort metrics.

## Create a Data Dictionary

A useful data dictionary includes:

- Column name.
- Type.
- Meaning.
- Unit.
- Allowed values.
- Missing-value meaning.
- Example.
- Source.

This is simple work, but it prevents many misunderstandings.

## Split the Dataset

Choose a split that matches reality:

- Random split for independent examples.
- Time-based split for future prediction.
- Group split when records from the same entity could leak.

Keep the test set untouched until final evaluation.

## Prevent Leakage

Watch for:

- Features computed using future information.
- Target-derived fields.
- Duplicates across train and test.
- Preprocessing fit before splitting.
- Aggregates that include the target period.

Leakage makes models look impressive in notebooks and disappointing in production.

## Preparation Checklist

Before analysis or modeling:

- Problem is defined.
- Target and prediction time are clear.
- Data sources are documented.
- Granularity is clear.
- Schema is understood.
- Split strategy matches the task.
- Leakage risks are reviewed.
- Privacy requirements are checked.

Data preparation is not glamorous, but it decides whether the analysis can be trusted.
