---
title: "Clustering: Apriori"
date: 2020/02/11
categories:
  - Blogs
tags:
  - Unsupervised Learning
  - Association Rules
excerpt: "A practical introduction to Apriori and association rule mining, including support, confidence, lift, frequent itemsets, and market-basket analysis."
layout: post
mathjax: true
toc: true
---

## Introduction

Apriori is not a clustering algorithm in the usual sense. It is an association rule mining algorithm. It finds itemsets that frequently appear together and derives rules from them.

Classic use case: market-basket analysis.

```text
Customers who buy bread and peanut butter also often buy jam.
```

## Frequent Itemsets

An itemset is a set of items, such as:

```text
{bread, peanut butter}
```

Support measures how often an itemset appears:

$$
support(A) = \frac{\text{transactions containing } A}{\text{all transactions}}
$$

Apriori finds itemsets whose support exceeds a minimum threshold.

## Apriori Principle

The key property:

```text
If an itemset is frequent, all of its subsets must also be frequent.
```

This lets the algorithm prune the search space. If `{bread, jam}` is not frequent, then `{bread, jam, milk}` cannot be frequent.

## Association Rules

A rule has the form:

```text
A -> B
```

Confidence measures how often B appears when A appears:

$$
confidence(A \to B) = \frac{support(A \cup B)}{support(A)}
$$

Lift compares the rule to chance:

$$
lift(A \to B) = \frac{confidence(A \to B)}{support(B)}
$$

Lift greater than 1 suggests A and B appear together more often than expected if independent.

## Practical Use

Apriori is useful for:

- Basket analysis.
- Product bundling.
- Recommendation rules.
- Event co-occurrence.
- Pattern discovery in transaction data.

Watch out:

- Too low support creates too many rules.
- High confidence can be misleading for very common items.
- Rules show association, not causation.
- Business usefulness matters more than rule count.

## Closing

Apriori is a clear method for discovering frequent co-occurrence patterns. It is best used as exploratory analysis or as a simple rule-based recommendation ingredient.
