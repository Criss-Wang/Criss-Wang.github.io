---
title: "Clustering: DBSCAN"
date: 2020/02/06
categories:
  - Blogs
tags:
  - Unsupervised Learning
  - Clustering
excerpt: "A practical guide to DBSCAN, density-based clustering, epsilon, minimum samples, noise points, and when density clustering is useful."
layout: post
toc: true
---

## Introduction

DBSCAN is a density-based clustering algorithm. It groups points that are packed closely together and marks isolated points as noise.

Unlike K-means, DBSCAN does not require choosing the number of clusters ahead of time.

## Core Ideas

DBSCAN uses two main parameters:

- `eps`: neighborhood radius.
- `min_samples`: minimum points needed to form a dense region.

Point types:

- **Core point:** has at least `min_samples` points within `eps`.
- **Border point:** reachable from a core point but not dense enough itself.
- **Noise point:** not reachable from dense regions.

## When DBSCAN Helps

Use DBSCAN when:

- Clusters have irregular shapes.
- Noise detection matters.
- Number of clusters is unknown.
- Density is meaningful in the feature space.

Examples:

- Geographic clusters.
- Sensor anomalies.
- Spatial event detection.

## Parameter Choice

The hardest part is choosing `eps`.

Common approach:

1. Compute distance to the k-th nearest neighbor.
2. Sort those distances.
3. Look for an elbow in the curve.

Feature scaling matters. If one feature dominates distance, DBSCAN will cluster mostly by that feature.

## Limitations

DBSCAN struggles when:

- Clusters have very different densities.
- Data is high-dimensional.
- Distance is not meaningful.
- `eps` is difficult to choose.

In high dimensions, consider dimensionality reduction or another method.

## Closing

DBSCAN is valuable because it finds dense regions and labels noise. It is a strong choice for spatial or density-shaped clustering problems, but it depends heavily on meaningful distances and good parameter choices.
