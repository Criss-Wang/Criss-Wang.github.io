---
title: "Stats in ML: Dirichlet Distribution"
date: 2021/07/21
categories:
  - Blogs
tags:
  - Probability
  - Machine Learning
excerpt: "A practical explanation of the Dirichlet distribution, its relationship to categorical probabilities, conjugacy with the multinomial, and why it appears in topic models."
layout: post
mathjax: true
toc: true
---

## Introduction

The Dirichlet distribution is a probability distribution over probability vectors. It is useful when the object you want to model is itself a set of probabilities that sum to one.

Examples:

- Topic proportions in a document.
- Class probabilities.
- Mixture weights.
- Category preferences.

If a categorical distribution chooses one category, a Dirichlet distribution can describe uncertainty over the categorical probabilities.

## Definition

For a vector $\theta = (\theta_1, \dots, \theta_K)$:

$$
\theta_i \ge 0,\quad \sum_{i=1}^{K}\theta_i = 1
$$

The Dirichlet distribution is parameterized by $\alpha = (\alpha_1, \dots, \alpha_K)$:

$$
\theta \sim Dirichlet(\alpha)
$$

Each $\alpha_i$ controls how much mass the distribution gives to category $i$.

## Intuition

Think of $\alpha_i$ as pseudo-counts.

- If all $\alpha_i$ are small, samples are sparse and often put most mass on a few categories.
- If all $\alpha_i$ are large, samples concentrate near the average proportions.
- If one $\alpha_i$ is larger than the others, samples tend to assign more probability to that category.

For a symmetric Dirichlet where all $\alpha_i = \alpha$:

- $\alpha < 1$: sparse probability vectors.
- $\alpha = 1$: roughly uniform over the simplex.
- $\alpha > 1$: probability vectors near uniform.

## Conjugacy

The Dirichlet is conjugate to the categorical and multinomial distributions. This means that if the prior over category probabilities is Dirichlet, and we observe category counts, the posterior is also Dirichlet.

If:

$$
\theta \sim Dirichlet(\alpha)
$$

and the observed category counts are:

$$
n = (n_1, \dots, n_K)
$$

then:

$$
\theta \mid n \sim Dirichlet(\alpha_1+n_1, \dots, \alpha_K+n_K)
$$

This is one reason the Dirichlet appears so often in Bayesian models.

## Connection to Topic Models

Latent Dirichlet Allocation (LDA) uses Dirichlet distributions in two important places:

- A document has a distribution over topics.
- A topic has a distribution over words.

Sparse Dirichlet priors encourage each document to focus on a small number of topics and each topic to focus on a subset of words.

This matches the intuition that most documents are about a few themes, not every possible theme equally.

## Practical Notes

Use a Dirichlet when:

- You need a distribution over category probabilities.
- Components must be nonnegative and sum to one.
- You want a Bayesian prior for categorical or multinomial probabilities.
- You are modeling mixture weights.

Be careful:

- Dirichlet components are negatively correlated because they must sum to one.
- The parameters are not direct probabilities.
- Very small concentration values can produce extremely sparse samples.

## Closing

The Dirichlet distribution is easiest to remember as a distribution over probability vectors. It gives a principled way to represent uncertainty about category proportions, especially when paired with multinomial observations.
