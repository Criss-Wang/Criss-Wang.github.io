---
title: "Recommender Systems II: Factorization Machines"
date: 2020/04/04
categories:
  - Blogs
tags:
  - Recommender Systems
  - Machine Learning
excerpt: "A practical explanation of factorization machines, feature interactions, sparse data, and why they are useful in recommendation and click prediction."
layout: post
mathjax: true
toc: true
---

## Introduction

Factorization machines (FMs) model interactions between sparse features. They are especially useful in recommendation and click-through-rate prediction, where inputs often contain many categorical features.

Examples:

- User ID.
- Item ID.
- Query.
- Category.
- Device.
- Context.

One-hot encoding creates huge sparse feature vectors. FMs handle this by learning low-dimensional embeddings for features and modeling pairwise interactions through those embeddings.

## Linear Model Limitation

A linear model predicts:

$$
\hat{y} = w_0 + \sum_i w_i x_i
$$

This captures individual feature effects but not interactions such as:

```text
user A likes item B
```

Adding every pairwise interaction manually is expensive because sparse data has many possible pairs.

## Factorization Machine Formula

A second-order FM predicts:

$$
\hat{y} =
w_0
+ \sum_i w_i x_i
+ \sum_i \sum_{j>i} \langle v_i, v_j \rangle x_i x_j
$$

Here $v_i$ is a learned embedding vector for feature $i$. The dot product $\langle v_i, v_j \rangle$ estimates the strength of interaction between two features.

This makes pairwise interactions possible without learning a separate parameter for every feature pair.

## Why It Works Well for Sparse Data

Sparse recommendation data rarely observes every user-item pair. FMs generalize by sharing information through embeddings.

If two items have similar embeddings, the model can transfer signal even when one item has fewer observations.

## Use Cases

FMs are useful for:

- Click-through-rate prediction.
- Recommendation ranking.
- Ad ranking.
- User-item interaction modeling.
- Sparse categorical data.

They are often used as a strong baseline before moving to deeper models.

## Extensions

Common extensions include:

- Field-aware factorization machines.
- DeepFM.
- Wide and deep models.
- Neural collaborative filtering.

The shared idea is to combine memorization of known feature effects with generalization through embeddings.

## Practical Notes

When using FMs:

- Encode categorical features carefully.
- Use regularization.
- Monitor rare categories.
- Compare against simple baselines.
- Evaluate ranking metrics, not only loss.

FMs are not magic, but they are a clean way to model sparse feature interactions.
