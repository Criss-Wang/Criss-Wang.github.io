---
title: "Recommender Systems III: Deep Learning Methods"
date: 2020/04/05
categories:
  - Blogs
tags:
  - Recommender Systems
  - Deep Learning
excerpt: "A practical overview of deep learning methods for recommender systems, including embeddings, two-tower retrieval, ranking models, sequence models, and evaluation."
layout: post
mathjax: true
toc: true
---

## Introduction

Deep learning is useful in recommender systems when raw features, sequence behavior, text, images, or large-scale embeddings can improve recommendations.

Modern recommender systems often have two stages:

1. Candidate generation: retrieve a small set of possible items.
2. Ranking: score and order those candidates.

Deep learning can help in both stages.

## Embeddings

Embeddings map users, items, and context features into dense vectors.

Examples:

- User embedding.
- Item embedding.
- Query embedding.
- Category embedding.
- Text or image embedding.

The model learns that similar users or items should have nearby representations.

## Two-Tower Retrieval

A two-tower model has one tower for users or queries and one tower for items.

```text
user/context -> user tower -> user embedding
item features -> item tower -> item embedding
```

Candidate retrieval uses similarity between embeddings, often dot product or cosine similarity.

Two-tower models are popular because item embeddings can be precomputed and searched efficiently with approximate nearest neighbor indexes.

## Ranking Models

Ranking models score a smaller candidate set with richer features.

Inputs may include:

- User features.
- Item features.
- Context.
- Historical behavior.
- Candidate source.
- Price or freshness.
- Cross features.

Ranking models can be gradient-boosted trees, wide-and-deep models, DeepFM-style models, transformers, or other neural architectures.

## Sequence Models

User behavior is sequential. The next item a user wants often depends on the recent session.

Sequence models use:

- Recent clicks.
- Recent purchases.
- Watch history.
- Search queries.
- Time gaps.

Transformers and recurrent models can represent session intent, but they are more expensive than simple embedding averages. Use them when sequence behavior clearly matters.

## Content Understanding

Deep learning also helps build item representations from content:

- Text descriptions.
- Images.
- Audio.
- Video.
- Reviews.

This is especially helpful for cold-start items because the system can recommend a new item before much interaction data exists.

## Evaluation

Offline metrics:

- Recall@k for candidate generation.
- nDCG for ranking.
- Precision@k.
- Coverage.
- Diversity.

Online metrics:

- Click-through rate.
- Conversion.
- Watch time.
- Retention.
- Revenue.
- User satisfaction.

Be careful: optimizing clicks alone can create shallow recommendations. Track long-term and quality guardrails.

## Closing

Deep recommender systems are powerful because they learn representations from users, items, context, and content. The engineering challenge is to keep retrieval fast, ranking accurate, and evaluation aligned with user value.
