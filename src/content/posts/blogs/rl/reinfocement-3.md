---
title: "Reinforcement Learning: Theoretical Foundations, Part III"
date: 2021/01/07
categories:
  - Blogs
tags:
  - Reinforcement Learning
  - Machine Learning
excerpt: "A practical guide to dynamic programming, policy evaluation, policy improvement, value iteration, and Q-learning."
layout: post
mathjax: true
toc: true
---

## Introduction

Once an RL problem is written as an MDP, we can ask how to find a good policy. Classical methods start with value functions and Bellman updates.

This post covers:

- Policy evaluation.
- Policy improvement.
- Value iteration.
- Q-learning.

## Policy Evaluation

Policy evaluation estimates how good a fixed policy is.

Given a policy $\pi$, estimate:

$$
V^\pi(s)
$$

by repeatedly applying the Bellman expectation update until values stabilize.

## Policy Improvement

If we know the value of states or actions, we can improve the policy by choosing better actions.

Greedy improvement chooses:

$$
\pi'(s) = \arg\max_a Q^\pi(s,a)
$$

Policy iteration alternates evaluation and improvement.

## Value Iteration

Value iteration combines evaluation and improvement into one update:

```text
new value = best expected immediate reward + discounted next-state value
```

It is useful when the transition model is known.

## Q-Learning

Q-learning learns action values from experience without needing a full transition model.

Its update can be read as:

```text
target = reward + gamma * best_next_q
Q(s, a) = Q(s, a) + alpha * (target - Q(s, a))
```

The difference between the target and the current Q-value is the temporal-difference error.

Q-learning is off-policy because it can learn the value of a greedy policy while behavior still explores.

## Exploration

A common exploration method is epsilon-greedy:

- With probability $\epsilon$, choose a random action.
- Otherwise, choose the action with highest current Q-value.

Over time, $\epsilon$ is often decayed.

## Closing

Value-based methods learn what states or actions are worth. They are foundational, but they become difficult when state or action spaces are large, continuous, or partially observed.
