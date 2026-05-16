---
title: "Reinforcement Learning: Theoretical Foundations, Part V"
date: 2021/01/20
categories:
  - Blogs
tags:
  - Reinforcement Learning
  - Machine Learning
excerpt: "A practical overview of actor-critic methods, deep reinforcement learning, replay buffers, target networks, PPO-style updates, and evaluation concerns."
layout: post
mathjax: true
toc: true
---

## Introduction

Actor-critic methods combine two ideas:

- An **actor** learns the policy.
- A **critic** estimates value.

The critic helps reduce variance in policy updates, while the actor learns how to act.

## Actor and Critic

The actor is:

$$
\pi_\theta(a \mid s)
$$

The critic estimates:

$$
V_\phi(s)
$$

or:

$$
Q_\phi(s,a)
$$

The actor uses the critic's estimates to improve the policy.

## Temporal-Difference Learning

The critic often learns from temporal-difference targets:

$$
r + \gamma V(s')
$$

The TD error is:

$$
\delta = r + \gamma V(s') - V(s)
$$

This can be used as an advantage estimate for actor updates.

## Deep RL Stabilizers

Deep RL adds neural networks, which makes function approximation powerful but unstable.

Common stabilizers:

- Replay buffers.
- Target networks.
- Advantage normalization.
- Entropy bonuses.
- Gradient clipping.
- Reward normalization.
- Clipped policy updates.

## PPO-Style Updates

Proximal Policy Optimization (PPO) limits how much the policy changes in one update. The goal is to improve the policy without taking destructive steps.

This is why PPO became a practical default in many RL settings: it balances simplicity and stability.

## Evaluation

RL evaluation is tricky.

Track:

- Average return.
- Success rate.
- Episode length.
- Variance across seeds.
- Safety violations.
- Sample efficiency.
- Performance under environment changes.

Always evaluate multiple random seeds. A single lucky run is not evidence.

## Practical Warning

RL is often overkill. Use it when actions influence future states and delayed reward matters. If the task is static prediction, supervised learning is usually simpler and better.

## Closing

Actor-critic methods are the bridge between value estimation and direct policy optimization. They are powerful, but their training dynamics require careful evaluation, multiple seeds, and strong baselines.
