---
title: "Reinforcement Learning: Theoretical Foundations, Part I"
date: 2021/01/04
categories:
  - Blogs
tags:
  - Reinforcement Learning
  - Machine Learning
excerpt: "A practical introduction to reinforcement learning concepts: agent, environment, state, action, reward, policy, return, and the exploration-exploitation tradeoff."
layout: post
mathjax: true
toc: true
---

## Introduction

Reinforcement learning (RL) studies how an agent learns to make decisions by interacting with an environment.

At each time step:

1. The agent observes a state.
2. The agent chooses an action.
3. The environment returns a reward and a new state.
4. The agent updates its behavior to get more reward over time.

## Core Terms

- **Agent:** the learner or decision maker.
- **Environment:** the world the agent interacts with.
- **State:** information available to the agent.
- **Action:** a decision the agent can take.
- **Reward:** feedback from the environment.
- **Policy:** a rule for choosing actions.
- **Return:** accumulated future reward.

The goal is to learn a policy that maximizes expected return.

## Return and Discounting

The discounted return is:

$$
G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \dots
$$

where $\gamma$ is the discount factor.

- $\gamma$ near 0 values immediate reward.
- $\gamma$ near 1 values long-term reward.

Discounting helps make long-horizon problems mathematically manageable and encodes how much future rewards matter.

## Exploration and Exploitation

The agent must balance:

- **Exploration:** try actions to learn more.
- **Exploitation:** choose actions that currently seem best.

Too little exploration can trap the agent in a poor policy. Too much exploration can waste reward.

## Why RL Is Hard

RL is difficult because:

- Rewards can be delayed.
- Data depends on the policy.
- Exploration can be expensive or unsafe.
- Training can be unstable.
- Simulators may not match reality.

RL is powerful, but it should not be used when supervised learning, planning, or simpler optimization would solve the problem.

## Closing

The foundation of RL is the agent-environment loop. Everything else, value functions, policy gradients, Q-learning, and actor-critic methods, builds on this loop.
