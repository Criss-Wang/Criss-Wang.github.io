---
title: "Dimensionality Reduction: Life Savers"
date: 2020/03/04
categories:
  - Blogs
tags:
  - Machine Learning
  - Feature Engineering
excerpt: "A practical guide to dimensionality reduction, including PCA, t-SNE, UMAP, autoencoders, feature selection, and how to choose the right method."
layout: post
mathjax: true
toc: true
---

## Introduction

Dimensionality reduction turns many features into fewer features while trying to preserve the information that matters.

It is useful when:

- Features are redundant.
- Models overfit high-dimensional noise.
- Training is too slow.
- Visualization is needed.
- Storage or inference cost is too high.
- Distances become unreliable in very high dimensions.

The goal is not always maximum compression. The goal is preserving useful signal while removing unnecessary complexity.

## Feature Selection vs Feature Extraction

There are two major families.

**Feature selection** keeps a subset of original features. This is easier to explain because the selected features still have original meaning.

Examples:

- Remove low-variance features.
- Remove highly correlated features.
- Select by mutual information.
- Use model-based importance.
- Use recursive feature elimination.

**Feature extraction** creates new features from the original ones. This can preserve more signal, but the new features are often harder to interpret.

Examples:

- PCA.
- SVD.
- t-SNE.
- UMAP.
- Autoencoders.

## PCA

Principal Component Analysis (PCA) finds orthogonal directions that explain the largest variance in the data.

Use PCA when:

- Features are numeric.
- Linear structure is a reasonable assumption.
- You want a fast baseline.
- You need a lower-dimensional representation for modeling or visualization.

Watch out:

- PCA is sensitive to scale, so standardize features first.
- Components are linear combinations and may be hard to explain.
- High variance does not always mean high predictive value.

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


x_scaled = StandardScaler().fit_transform(x)
x_reduced = PCA(n_components=20).fit_transform(x_scaled)
```

## t-SNE and UMAP

t-SNE and UMAP are often used for visualization. They preserve local neighborhood structure better than PCA, but they are not usually the first choice for production features.

Use them for:

- Exploring clusters.
- Visualizing embeddings.
- Inspecting representation quality.
- Communicating high-dimensional structure.

Be careful:

- Axes do not have simple meaning.
- Distances between far-away clusters can be misleading.
- Hyperparameters can change the visual story.
- A pretty plot is not evidence of model quality.

## Autoencoders

An autoencoder is a neural network trained to reconstruct its input through a compressed bottleneck. The bottleneck representation can be used as a learned low-dimensional feature.

Use autoencoders when:

- The data is nonlinear.
- You have enough data to train a neural representation.
- Reconstruction is related to the downstream task.
- You can evaluate whether the learned representation helps.

They are more flexible than PCA but require more engineering and validation.

## Choosing a Method

Use this practical decision path:

1. If interpretability matters, start with feature selection.
2. If you need a fast numeric baseline, start with PCA.
3. If you need visualization, try PCA first, then UMAP or t-SNE.
4. If the data is nonlinear and large enough, consider autoencoders.
5. If the downstream model is tree-based, check whether dimensionality reduction is needed at all.

Always compare downstream performance with and without reduction. Dimensionality reduction can remove noise, but it can also remove signal.

## Evaluation

Evaluate dimensionality reduction by:

- Downstream metric.
- Reconstruction error, when relevant.
- Stability across random seeds.
- Interpretability.
- Runtime.
- Memory reduction.
- Slice-level performance.

For visualization, pair plots with quantitative checks. Do not make a product decision from a 2D embedding alone.

## Closing

Dimensionality reduction is useful when complexity is hurting the system. It should simplify modeling, visualization, storage, or inference without hiding important signal.

Treat it as an engineering tradeoff, not as automatic preprocessing.
