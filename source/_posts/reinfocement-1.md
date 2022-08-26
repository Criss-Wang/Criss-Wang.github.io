---
title: "Reinforcement Learning - Theoretical Foundations: Part I "
excerpt: "Finally the most intriguing part (to me)"
date: 2021/01/04
updated: 2022/8/19
categories:
  - Blogs
tags: 
  - Reinforcement Learning
layout: post
mathjax: true
toc: true
---
### Introduction
Recently I\'ve been learning about reinforcement learning from amazing [lectures from David Silver](https://www.davidsilver.uk/teaching/). These provide an overview of the classical algorithms in RL and potential challenges for future researches, in the subsequent blogs, I\'ll talk about the major aspects of RL and provide some solid math details on how algorithms in RL is executed. 

### What is Reinforcement Learning
An RL agent may include one or more of these components:
- Policy: agent’s behaviour function
  - It is a map from state to action
  - Deterministic policy: a = $\pi(s)$
  - Stochastic policy: $\pi(a\|s) = P[A_t = a\|S_t = s]$
- Value function: how good is each state and/or action
  - Value function is a prediction of future reward
  - Used to evaluate the goodness/badness of states
  - And therefore to select between actions
- Model: agent’s representation of the environment
  - A model predicts what the environment will do next
  - $P$ predicts the next state
  - $R$ predicts the next (immediate) reward, this often takes the form of expectation

It is derived for a classical problem - Exploration vs Exploitation. There is no supervisor, only a reward signal. Sometimes, feedback is delayed, not instantaneous. Time really matters (sequential, non i.i.d data); and Agent\'s actions affect the subsequent data it receives.

### Prediction vs Control
RL problems is often classified into a prediction problem or a control problem
- Prediction: **Given a policy**, evaluate the future
- Control: **Find the best policy**

### Markov Decision Process (MDP)
Before venturing into the exact algorithms, let\'s lay out some fundamental math concepts here.

#### Prior Knowledge
- Basic Probability Knowledge
- Basic Stochastic Process:
	- Markov Chain
	- Transition Matrix

### Problem setup
1. This is an RL setting where the environment is fully observable
2. The current *state* completely characterises the process
3. Almost all RL problems can be formalised as MDPs (Bandit are MDP with 1 state & finite/infinite actions)

### Terminologies
1. **Markov Reward Process**
  - A Markov reward process is a Markov chain with values.
  - In addition to $\langle S, T, P\rangle$, we have reward function $\cal{R}\_k = \mathbb{E}[R_{t+1}\\|S_t = k]$ and a discount factor $\gamma \in [0,1]$
2. **Return**
  - The return $G_t$ is the total discounted reward from time-step $t$: 
  - $G_t = \sum_{k = 0}^{\infty} \gamma^k R_{t+k+1}$
  - Intuitively, the discount factor $\gamma$ favours future rewards at a nearer date
3. **State-value function**
- The state value function $v(s)$ of an MRP is the expected return starting from state $s$
- $v(s) = \mathbb{E}[G_t \| S_t = s]$
4. **Bellman Equation**
- We apply one-step analysis on $v(s)$ and observe that: $v(s) = \mathbb{E}[R_{t+1} + \gamma v(S_{t+1}) \| S_t = s]$
- Therefore, we find that value function of any state is only dependent on its outgoing states (successors)
- We can then construct a matrix equation: $V = {\cal{R}} + \gamma {\cal P} V$, which can be solved in $O(n^3)$ (hence very expensive)
- For large state set, more efficient algorithms utilising Dynamic programming (DP) is preferred
5. **MDP**
- A Markov decision process (MDP) is a Markov reward process with decisions
- In addition to states, we have a set of actions $\cal A$
- The probability and reward functions now conditionally depend on $S_t$ and $A_t$ simultaneous as follows $f(\cdot \| S_t = s, A_t = a)$
6. **Policy**
- A policy $π$ is a distribution over actions given states: $\pi(a\|s) = {\mathbb{P}}[A_t = a \| S_t = s]$
- A policy is stationary (the probability does not change for different iterations)
7. **State-value function and Action-value function for MDP** (Differs from 3.)
- State-value function: $v_{\pi}(s) = \mathbb{E_{\pi}}[G_t \| S_t = s]$ $\implies$ the expectation taken w.r.t the policy $\pi$
- Action-value function: $q_{\pi}(s,a) = \mathbb{E_{\pi}}[G_t \| S_t = s, A_t = a]$
- Note that $v_{\pi}(s) = q_{\pi}(s,\pi(s))$
8. **Applying Bellman equation** on $v_{\pi}(s)$ and $q_{\pi}(s,a)$
- Note that we cannot do simple one-step analysis since there are 2 variables $S_t$ and $A_t$ now.
- Instead, we try to apply OSA on $S_t$ w.r.t $A_t$, and then do OSA on $A_t$ w.r.t $S_t$ to get Bellman Expectation Equation on $S_t$, and swap the 2 variables' order to derive equation for $A_t$
- First step (Bellman Expectation Equation): 
 - $v_{\pi}(s) = \mathbb{E_{\pi}}[R_{t+1} + \gamma v_{\pi}(S_{t+1})\| S_t = s] = \sum_{a\in{\cal A}}\pi(a\|s)q_{\pi}(s,a)$ 
 - $q_{\pi}(s,a) = \mathbb{E_{\pi}}[ R_{t+1} + \gamma q_{\pi}(S_{t+1}, A_{t+1}) \| S_t = s, A_t = a]={\cal R}^a_s + \gamma \sum_{s'\in S}{\cal P}^a_{ss'}v_{\pi}(s')$
- Second step (Bellman Optimality Equation):
-  $v_{\pi}(s) =  \sum_{a\in{\cal A}}\pi(a\|s) ( {\cal R}^a_s + \gamma \sum_{s'\in S}{\cal P}^a_{ss'}v_{\pi}(s'))$
- $q_{\pi}(s,a) ={\cal R}^a_s + \gamma \sum_{s'\in S}{\cal P}^a_{ss'}\sum_{a'\in{\cal A}}\pi(a'\|s')q_{\pi}(s',a')$
9. **Optimality**
- We try to maximize  $v_{\pi}(s)$ and $q_{\pi}(s,a)$ (Goal 1)
- Theorem suggests existence of policy $\pi_*$that achieves Goal 1 deterministically.
- An optimal policy can be found by choosing actions $a$ for each state $s$ such that the choices maximise over $q_{\pi}(s,a)$
10. **Solving for optimality**
- The Bellman equations in 8 is often nonlinear and has no closed form in general
- We often need to apply sequential methods to solve it
 - Value iteration
 - Policy Iteration
 - Q-Learning
 - Sarsa
11. To read more on extensions, refer to Page 49 of [this slides](https://www.davidsilver.uk/wp-content/uploads/2020/03/MDP.pdf).


