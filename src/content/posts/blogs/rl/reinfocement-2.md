---
title: "Reinforcement Learning: Theoretical Foundations, Part II"
date: 2021/01/05
categories:
  - Blogs
tags:
  - Reinforcement Learning
  - Machine Learning
excerpt: "A practical explanation of Markov decision processes, transition dynamics, rewards, policies, value functions, and Bellman equations."
layout: post
mathjax: true
toc: true
---

## Introduction

The Markov decision process (MDP) is the standard mathematical framework for reinforcement learning.

An MDP describes:

- States.
- Actions.
- Transition probabilities.
- Rewards.
- Discount factor.

## Markov Property

The Markov property says the future depends on the current state and action, not the full past history:

$$
P(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \dots)
=
P(s_{t+1} \mid s_t, a_t)
$$

This assumption lets us write efficient algorithms.

## Policy

A policy maps states to actions.

A deterministic policy:

$$
a = \pi(s)
$$

A stochastic policy:

$$
\pi(a \mid s) = P(a_t = a \mid s_t = s)
$$

The agent's goal is to find a policy with high expected return.

## Value Function

The state-value function is the expected return from state $s$ under policy $\pi$:

$$
V^\pi(s) = E_\pi[G_t \mid s_t=s]
$$

The action-value function is:

$$
Q^\pi(s,a) = E_\pi[G_t \mid s_t=s, a_t=a]
$$

Value functions estimate how good states or actions are.

## Bellman Equation

The Bellman equation expresses value recursively. In words:

```text
value now = expected immediate reward + discounted value later
```

A compact notation is:

$$
V^\pi(s) = E_\pi[r_{t+1} + \gamma V^\pi(s_{t+1}) \mid s_t=s]
$$

This recursion is the backbone of many RL algorithms. It says that a state's value can be estimated by looking one step ahead and then reusing the value estimate for the next state.

## Closing

MDPs give reinforcement learning its structure. Once states, actions, rewards, and transitions are defined, learning becomes the problem of estimating values or improving policies.
