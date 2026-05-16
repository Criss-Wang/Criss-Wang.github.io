---
title: "Unsupervised Learning: Measures About Clustering"
date: 2020/02/15
categories:
  - Blogs
tags:
  - Unsupervised Learning
  - Clustering
excerpt: "A practical guide to evaluating clustering with silhouette score, Davies-Bouldin index, Calinski-Harabasz score, external labels, stability, and business usefulness."
layout: post
mathjax: true
toc: true
---

## Introduction

Unsupervised learning finds structure without target labels. Clustering is one of the most common unsupervised tasks: grouping similar examples together.

The challenge is evaluation. Without labels, there is no obvious "accuracy."

## Internal Metrics

Internal metrics evaluate clusters using only the input data and cluster assignments.

### Silhouette Score

Silhouette score compares how close a point is to its own cluster versus other clusters.

Values range from -1 to 1:

- Near 1: well matched to its cluster.
- Near 0: near a boundary.
- Negative: possibly assigned to the wrong cluster.

### Davies-Bouldin Index

Davies-Bouldin measures average cluster similarity. Lower is better.

It rewards compact clusters that are far apart.

### Calinski-Harabasz Score

Calinski-Harabasz compares between-cluster dispersion to within-cluster dispersion. Higher is better.

It often favors well-separated dense clusters.

## External Metrics

If labels are available for evaluation, use external metrics:

- Adjusted Rand Index.
- Normalized Mutual Information.
- Homogeneity.
- Completeness.
- V-measure.

These labels do not need to be training labels. They can be human categories or downstream outcomes used only for evaluation.

## Stability

A clustering result should be stable enough to trust.

Check:

- Different random seeds.
- Different samples of the data.
- Small perturbations.
- Different feature sets.
- Different numbers of clusters.

If clusters change completely with tiny changes, the structure may be weak.

## Interpretability

A clustering result should be explainable:

- What defines each cluster?
- Which features separate clusters?
- Are clusters actionable?
- Are clusters stable over time?
- Do domain experts recognize them?

Pretty 2D plots are not enough. Useful clusters should support decisions.

## Practical Checklist

Before using clusters:

- Standardize features when distance matters.
- Remove leakage or ID-like features.
- Try simple baselines.
- Compare multiple metrics.
- Inspect cluster profiles.
- Check stability.
- Validate usefulness with downstream tasks or domain review.

Clustering is exploratory. Treat it as a way to generate structure and hypotheses, not automatic truth.
