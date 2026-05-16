---
title: "Recommender Systems I: Content-Based and Collaborative Filtering"
date: 2020/04/01
categories:
  - Blogs
tags:
  - Recommender Systems
  - Machine Learning
excerpt: "A practical introduction to content-based recommendation, collaborative filtering, user-item matrices, similarity, cold start, and evaluation."
layout: post
mathjax: true
toc: true
---

## Introduction

Recommender systems help users discover items: movies, products, songs, posts, jobs, restaurants, or documents.

Two classic families are:

- Content-based recommendation.
- Collaborative filtering.

Modern systems often combine both, but the distinction is still useful.

## Content-Based Recommendation

Content-based systems recommend items similar to what a user already liked.

They rely on item features:

- Movie genre.
- Product category.
- Text description.
- Tags.
- Price.
- Brand.
- Embeddings.

The system builds a user profile from previously liked or consumed items, then recommends items with similar features.

### Pros

- Works for new items if item features exist.
- Easier to explain.
- Does not require many users.

### Cons

- Can become repetitive.
- Depends on feature quality.
- Struggles to discover surprising items outside the user's history.

## Collaborative Filtering

Collaborative filtering uses behavior patterns across users and items.

The core idea:

```text
Users who behaved similarly in the past may like similar items in the future.
```

The data is often represented as a user-item matrix:

```text
rows: users
columns: items
values: ratings, clicks, purchases, watch time, or implicit feedback
```

### User-Based Collaborative Filtering

Find users similar to the target user, then recommend items those similar users liked.

### Item-Based Collaborative Filtering

Find items similar to items the user liked, then recommend those items.

Item-based methods are often more stable because item similarities can change more slowly than user preferences.

## Similarity

Common similarity measures:

- Cosine similarity.
- Pearson correlation.
- Jaccard similarity.
- Dot product of embeddings.

The right measure depends on whether feedback is explicit ratings, implicit actions, binary events, or dense embeddings.

## Cold Start

Cold start happens when the system lacks history.

New users:

- Ask onboarding questions.
- Use popularity or trending items.
- Use contextual signals.
- Use content-based recommendations.

New items:

- Use item metadata.
- Use text or image embeddings.
- Explore with limited traffic.

Hybrid systems often exist because pure collaborative filtering struggles with cold start.

## Evaluation

Offline metrics:

- Precision@k.
- Recall@k.
- nDCG.
- Mean reciprocal rank.
- Coverage.
- Diversity.

Online metrics:

- Click-through rate.
- Conversion.
- Watch time.
- Retention.
- Revenue.
- User satisfaction.

A recommender can improve clicks while hurting long-term trust. Always use guardrail metrics.

## Closing

Content-based methods use item meaning. Collaborative filtering uses collective behavior. Strong recommender systems usually combine both and evaluate beyond a single ranking metric.
