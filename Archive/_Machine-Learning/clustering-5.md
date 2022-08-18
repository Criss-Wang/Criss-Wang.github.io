---
title: "Clustering: Apriori"
date: 2020-02-11
layout: single
author_profile: true
categories:
  - Unsupervised Learning
tags: 
  - Clustering
  - Bayesian Inference
excerpt: "Association Rule realized via inference"
mathjax: "true"
---
## Association Rule
Association rule learning is a rule-based machine learning method for discovering interesting relations between variables in large databases. It is intended to identify strong rules discovered in databases using some measures of interestingness. For example, we may want to find 1-1 product category assocaition rule: product cateogry 1 -> product category 2

This is often used for discovering regularities between products in large-scale transaction data recorded by point-of-sale (POS) systems in supermarkets. Because we don't have initial associations in our data, it is an unsupervised learning problem for marketing activities such as, e.g., promotional pricing or product placements. In contrast with sequence mining, association rule learning typically does not consider the order of items either within a transaction or across transactions. [Wikipedia]
## Evaluation Metrics [^1]
[^1]: https://michael.hahsler.net/research/recommender/associationrules.html
1. Support
  - $P(X,Y)$: % of transactions where items in X AND Y are bought together
  - Property of down-ward closure which means that all sub sets of a frequent set (support > min. support threshold) are also frequent
  - Cons: Items that occur very infrequently in the data set are pruned although they would still produce interesting and potentially valuable rules.
2. Confidence
  - $P(Y\|X)$: % of transactions amongst all customers who bought Y given that they have bought X
  - While support is used to prune the search space and only leave potentially interesting rules, confidence is used in a second step to filter rules that exceed a min. confidence threshold
  - Cons: sensitive to the frequency of the consequent (Y) in the data set. Caused by the way confidence is calculated, Ys with higher support will automatically produce higher confidence values even if they exists no association between the items.
3. Lift
  - $\frac{P(X,Y)}{P(X)P(Y)} = \frac{P(Y\|X)}{P(Y)}$
  - An association rule X -> Y is only useful if the lift value > 1
  - Want to consider also the presence of Y being bought independently without knowledge about X
  - Largely solves to problem of confidence threshold: sensitive to the frequency of the consequent (Y)
4. Conviction
  - $\frac{P(X)P(\neg Y)}{P(X, \neg Y)} = \frac{1-P(Y)}{1-P(Y\|X)}$: How poor can the association be.
  - A directed measure  monotone in confidence and lift.
5. Leverage
  - $P(X,Y) - P(X)P(Y)$: difference of X and Y appearing together in the data set and what would be expected if X and Y where statistically independent.
  - The rational in a sales setting is to find out how many more units (items X and Y together) are sold than expected from the independent sells.
  - Cons: suffer from the rare item problem.

## Apriori Property
> All subsets of a frequent itemset must be frequent(Apriori propertry). If an itemset is infrequent, all its supersets will be infrequent.

Applying the apriori property, we get the following algorithm.

### Algorithm
1. Generating **Support** Value for Itemsets containing one items (*One Itemset*)
2. With a pre-defined **support** threshold, identify itemsets worth exploring
3. With the shortlisted *One Itemset* that are above the **support** threshold, generate Itemsets containing two items (*Two Itemsets*)
4. With the same pre-definited **support** threshold, identify associations in *Two Itemsets* that are worth exploring
5. With the shortlisted *Two Itemsets*, association rule is generated between the two items
6. Confidence value is generated for each association rule
7. With a pre-defined **confidence** threshold, association rules are being shortlisted
8. With shortlisted association rules, the lift values are computed for each of them
9. Only association rules with lift value > 1 is considered as meaningful associations