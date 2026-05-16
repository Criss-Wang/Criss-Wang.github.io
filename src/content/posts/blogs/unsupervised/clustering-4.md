---
title: "Clustering: Affinity Propagation"
date: 2020/02/09
categories:
  - Blogs
tags:
  - Unsupervised Learning
  - Clustering
excerpt: "A concise explanation of affinity propagation, exemplars, similarity messages, preferences, damping, and practical limitations."
layout: post
toc: true
---

## Introduction

Affinity propagation is a clustering algorithm that identifies representative examples, called exemplars, and assigns other points to them.

Unlike K-means, it does not require specifying the number of clusters directly. Instead, the number of clusters is influenced by similarity values and preference settings.

## Core Idea

Affinity propagation passes messages between points:

- Responsibility: how well-suited a point is to be another point's exemplar.
- Availability: how appropriate it would be for a point to choose another point as exemplar.

Through iterative updates, exemplars emerge.

## Preference

The preference parameter controls how likely points are to become exemplars.

- Higher preference usually creates more clusters.
- Lower preference usually creates fewer clusters.

Choosing preference is one of the main practical tuning steps.

## Damping

Damping helps stabilize message updates. Without damping, the algorithm can oscillate and fail to converge.

## When to Use It

Affinity propagation can be useful when:

- You want representative examples for clusters.
- You do not know the number of clusters.
- The dataset is not too large.
- A meaningful similarity matrix is available.

## Limitations

Watch out:

- Memory cost can be high because pairwise similarities are needed.
- Runtime can be high on large datasets.
- Results can be sensitive to preference and damping.
- It is less common in production than K-means, DBSCAN, or hierarchical clustering.

## Closing

Affinity propagation is interesting because it chooses exemplars instead of abstract centroids. It is useful for certain exploratory tasks, but it needs careful tuning and does not scale as easily as simpler methods.
