---
title: "Clustering: K-Means and Gaussian Mixture Models"
date: 2020/02/01
categories:
  - Blogs
tags:
  - Unsupervised Learning
  - Clustering
excerpt: "A practical comparison of K-means and Gaussian mixture models, including assumptions, distance, soft assignments, initialization, and evaluation."
layout: post
mathjax: true
toc: true
---

## Introduction

K-means and Gaussian mixture models (GMMs) are two classic clustering methods. Both try to summarize data with groups, but they make different assumptions.

K-means assigns each point to one cluster center. GMMs model data as a mixture of Gaussian distributions and produce soft probabilities for cluster membership.

## K-Means

K-means tries to minimize within-cluster squared distance:

$$
\sum_{i=1}^{n} \|x_i - \mu_{c_i}\|^2
$$

where $\mu_{c_i}$ is the centroid of the assigned cluster.

Use K-means when:

- Clusters are roughly spherical.
- Features can be scaled meaningfully.
- Hard assignments are acceptable.
- You need a fast baseline.

Watch out:

- Sensitive to feature scale.
- Sensitive to initialization.
- Requires choosing $k$.
- Struggles with non-spherical clusters.
- Struggles with clusters of very different density.

## Gaussian Mixture Models

A GMM assumes data comes from a mixture of Gaussian components:

$$
p(x) = \sum_{k=1}^{K} \pi_k N(x \mid \mu_k, \Sigma_k)
$$

Each point receives a probability of belonging to each component.

Use GMMs when:

- Soft cluster membership is useful.
- Clusters may have elliptical shapes.
- You want a probabilistic model.

Watch out:

- More parameters than K-means.
- Sensitive to initialization.
- Can overfit with too many components.
- Covariance choices matter.

## Choosing the Number of Clusters

Useful tools:

- Elbow method.
- Silhouette score.
- BIC or AIC for GMMs.
- Stability checks.
- Domain interpretability.

The best number of clusters is not only a metric decision. It should also be useful for the downstream task.

## Closing

K-means is a fast hard-clustering baseline. GMMs are a probabilistic soft-clustering extension. Use both with scaling, stability checks, and careful interpretation.
