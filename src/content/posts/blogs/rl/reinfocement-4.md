---
title: "Reinforcement Learning: Theoretical Foundations, Part IV"
date: 2021/01/28
categories:
  - Blogs
tags:
  - Reinforcement Learning
  - Machine Learning
excerpt: "A practical introduction to policy-gradient methods, stochastic policies, the policy-gradient objective, baselines, and variance reduction."
layout: post
mathjax: true
toc: true
---

## Introduction

Value-based methods learn how good actions are. Policy-gradient methods directly optimize the policy.

They are useful when:

- Actions are continuous.
- A stochastic policy is desired.
- The policy is naturally parameterized by a neural network.

## Policy Objective

Let a policy be parameterized by $\theta$:

$$
\pi_\theta(a \mid s)
$$

The goal is to maximize expected return:

$$
J(\theta) = E_{\pi_\theta}[G_t]
$$

Policy-gradient methods estimate:

$$
\nabla_\theta J(\theta)
$$

and update the policy in the direction of higher expected return.

## REINFORCE

The classic Monte Carlo policy-gradient update is:

$$
\nabla_\theta J(\theta)
\approx
G_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)
$$

Intuition:

- If an action led to high return, increase its probability.
- If an action led to low return, decrease its probability.

## Baselines

Policy-gradient estimates can have high variance. A baseline reduces variance without changing the expected gradient.

Commonly:

$$
G_t - b(s_t)
$$

where $b(s_t)$ might be a value function estimate.

This leads toward actor-critic methods.

## Advantages

An advantage function measures how much better an action is than expected:

$$
A(s,a) = Q(s,a) - V(s)
$$

Using advantages helps the policy update focus on actions that are better or worse than the state's baseline expectation.

## Practical Issues

Policy-gradient methods can be unstable because updates may change the policy too much.

Common techniques:

- Advantage normalization.
- Entropy regularization.
- Gradient clipping.
- Trust-region or clipped objectives.
- Careful reward scaling.

## Closing

Policy gradients directly optimize behavior. They are flexible, especially for continuous control, but they need variance reduction and careful training discipline.
