---
date: 2026-05-05
layout: post
title: "Heron Event-Driven MARL"
categories:
  - Projects
  - Python
  - Multi-Agent RL
  - Event-driven Simulation
  - Agent Systems
excerpt: "A domain-agnostic MARL framework for evaluating trained policies under heterogeneous event-driven execution and realistic observability constraints"
link: "/images/Projects/heron-marl-diagram.svg"
mathjax: true
toc: true
---

### Introduction

HERON is a private domain-agnostic multi-agent reinforcement learning framework for hierarchical control systems where deployment timing is not clean, synchronized, or fully observable.

Most MARL policies are trained under `env.step()`, where every agent observes, acts, and updates in lockstep. Real systems behave differently: sensors tick at different rates, observations arrive late, and actuators respond after delay. HERON keeps the same agent definitions across step-based training and event-driven evaluation, so the deployment gap can be measured instead of guessed.

### What it focuses on

- Heterogeneous agent schedules with delay and jitter controls
- Declarative feature visibility for realistic observability
- Pluggable coordination protocols for hierarchical agents
- Dual-mode execution for step-based training and event-driven testing
- Power-grid and built-in demo environments for policy stress testing

### Why I built it

The point of HERON is to make asynchronous deployment failures attributable. Instead of treating performance drops as one vague sim-to-real gap, the framework lets experiments isolate timing, observability, and coordination structure as separate causes.
