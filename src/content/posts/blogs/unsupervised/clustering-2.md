---
title: "Clustering: Hierarchical, BIRCH, and Spectral"
date: 2020/02/05
categories:
  - Blogs
tags:
  - Unsupervised Learning
  - Clustering
excerpt: "A practical overview of hierarchical clustering, BIRCH, and spectral clustering, with guidance on when each method is useful."
layout: post
mathjax: true
toc: true
---

## Introduction

Not every clustering problem fits K-means. Hierarchical clustering, BIRCH, and spectral clustering offer different tradeoffs.

Use these methods when the structure is nested, large-scale, or not well represented by spherical clusters.

## Hierarchical Clustering

Hierarchical clustering builds a tree of clusters.

Two main styles:

- **Agglomerative:** start with each point as its own cluster, then merge.
- **Divisive:** start with one cluster, then split.

The result is often visualized as a dendrogram.

Common linkage choices:

- Single linkage.
- Complete linkage.
- Average linkage.
- Ward linkage.

Use hierarchical clustering when:

- You want nested cluster structure.
- The dataset is not too large.
- Interpretability is important.

Watch out for computational cost on large datasets.

## BIRCH

BIRCH builds a compact summary of the data using clustering features, then clusters the summaries.

It is useful for large datasets because it avoids storing every point in memory during clustering.

Use BIRCH when:

- The dataset is large.
- You need an efficient clustering summary.
- Incremental processing is useful.

It works best when clusters are reasonably compact.

## Spectral Clustering

Spectral clustering builds a similarity graph, uses eigenvectors of a graph Laplacian, and clusters in the transformed space.

It is useful for non-convex cluster shapes where distance-to-centroid methods fail.

Use spectral clustering when:

- Similarity graph structure matters.
- Clusters are not spherical.
- Dataset size is manageable.

Watch out:

- Similarity graph construction can be expensive.
- Choice of similarity scale matters.
- Large datasets can be difficult.

## Choosing Among Them

Use:

- **Hierarchical clustering** for nested structure and interpretability.
- **BIRCH** for scalable clustering summaries.
- **Spectral clustering** for graph-shaped or non-convex clusters.

Always inspect cluster stability and usefulness. The algorithm can produce clusters even when the structure is weak.
