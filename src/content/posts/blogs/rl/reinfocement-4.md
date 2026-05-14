---
title: "Reinforcement Learning - Theoretical Foundations: Part IV "
excerpt: "RL Continued - Value Function Approximation"
date: 2021/01/28
updated: 2022/01/14
categories:
  - Blogs
tags: 
  - Reinforcement Learning
layout: post
mathjax: true
toc: true
---
### Value Function Approximation

We know various methods can be applied for function approximation. For this note, we will mainly consider those differentiable methods: Linear Approximation and Neural Nets

#### 1. Stochastic Gradient Descent (SGD)
Here let\'s review a basic approximation strategy for gradient-based method: Stochastic Gradient Descent.

First our aim is to minimize the mean square error (MSE) between our estimator and the true function. The error is represented by 

$$J(\textbf{w}) = \mathbb{E}\_{\pi}[(\hat{v}(S, \textbf{w}) - v_{\pi}(S))^2].$$

To attain $\arg\min\limits\_{\textbf{w}} J(\textbf{w})$ we need to update the gradient until convergence.

A full gradient update 

$$\Delta(\textbf{w}) = \alpha\mathbb{E}\_{\pi}[(v_{\pi}(S) - \hat{v}(S, \textbf{w}) )\nabla\_{\textbf{w}}\hat{v}(S, \textbf{w})]$$ 

has the issue of converging at local minimum. Hence stochastic sampling with $\Delta(\textbf{w}) = \alpha(v_{\pi}(S) - \hat{v}(S, \textbf{w}))\nabla_{\textbf{w}}\hat{v}(S, \textbf{w})$ will work better in general.

#### 2. Linearization
We begin by considering a linear model. So $\hat{v}(S, \textbf{w}) = \textbf{x}(S)^T\textbf{w}$ where $\textbf{x}(S)$ is the feature vector/representation of the current state space. The stochastic update $\nabla_{\textbf{w}}\hat{v}(S, \textbf{w})$ in SGD is also updated to $\textbf{x}(S)$. 

On the other hand, we don\'t have an oracle for a known $v_{\pi}(S)$ in practice, so we need ways to estimate it. This is where algorithm design comes in.

### Algorithm analysis

#### 1. linear Monte-Carlo policy evaluation
- To represent $v_{\pi}(S_t)$, we use $G_t$. In every epoch, we apply supervised learning to “training data”: $\langle S_1, G_1\rangle , \langle S_2, G_2\rangle ,..., \langle S_T, G_T\rangle$.
- The update is now $\Delta(\textbf{w}) = \alpha(G_t - \hat{v}(S_t, \textbf{w}))\textbf{x}(S_t)$
- Note that Monte-Carlo evaluation converges to a local optimum 
- As $G_t$ is unbiased, it works even when using non-linear value function approximation

#### 2. TD Learning
- We use $R_{t+1} + \gamma \hat{v}(S_{t+1}, \textbf{w})$ for $v_{\pi}(S_t)$.
- TD(0) has the update formula: $\Delta(\textbf{w}) = \alpha(R_{t+1} + \gamma \hat{v}(S_{t+1}, \textbf{w}) - \hat{v}(S_t, \textbf{w}))\textbf{x}(S_t)$
- Linear TD(0) converges (close) to global optimum
- On the other hand we can use $\lambda$-return $G^{\lambda}_{t}$ as substitute. This is a TD($\lambda$) method.
- Forward view linear TD($\lambda$):  $\Delta(\textbf{w}) = \alpha(G^{\lambda}_{t} - \hat{v}(S_t, \textbf{w}))\textbf{x}(S_t)$
- Backward view linear TD($\lambda$) requires eligibility trace:
	- $\delta_t  = R_{t+1} + \gamma \hat{v}(S_{t+1}, \textbf{w}) - \hat{v}(S_t, \textbf{w})$
	- $E_t = \gamma\lambda E_{t-1} + \textbf{x}(S_t)$
	- $\Delta_{\textbf{w}} = \alpha\delta_tE_t$

#### 3. Convergence of Prediction Algorithms

| On\Off-policy  | Algorithm | Table-lookup | Linear | Non-Linear |
|--|--|--|--|--|
|On-Policy| MC | Y | Y | Y |
|On-Policy  | TD(0) | Y | Y | N |
|On-Policy| TD($\lambda$) | Y | Y | N |
|Off-Policy| MC | Y | Y | Y |
|Off-Policy  | TD(0) | Y | N | N |
|Off-Policy| TD($\lambda$) | Y | N | N |




## Action-Value Function Approximation
Now we don\'t simply approximate a value function $v_{\pi}(s)$, but approximate action-value function $q_{\pi}(s,a)$ instead.

The main idea is just find $\hat{q_{\pi}}(s, a, \textbf{w}) \approx q_{\pi}(s,a)$. Both MC and TD work the same way exactly by substituting these items inside the expressions.

## Improvements

### Gradient TD
Some more recent improves aim to resolve the failure of convergence of off-policy TD algorithms. This gave birth to a Gradient TD algorithm that converges in both linear and non-linear cases. This requires an additional parameter $\textbf{h}$ to be added and tuned which reprsents the gradient of projected Bellman error. In a similar fashion, a gradient Q-learning is also invented, but with no gurantee on non-linear model convergence.

### Least Squares Prediction and Experience Replay
LS estimator is known to approximate $\textbf{w}$ well in general. So instead of correctly approximating $\textbf{w}$, it may also be ideal to approximate $LS(\textbf{w})$ instead.

It is found that SGD with Experience Replay converges in this case. By \"Experience Replay\" we are storing the history in each epoch instead of discarding them after each iteration.  And we randomly selection some of these \"data\" for stochastic update in SGD.

### Deep Q-Networks (DQN)
- DQN uses **experience replay** and **fixed Q-targets**
- It takes actions based on a $\epsilon$-greedy policy
- Store transition $(s_t , a_t ,r_{t+1},s_{t+1})$ in replay memory $\cal D$ (experience replay)
- Sample random mini-batch of transitions from $\cal D$ 
- Compute Q-learning targets w.r.t. old, fixed parameters $\textbf{w}^-$ (fixed Q-target: not the latest $\textbf{w}$ but a $\textbf{w}$ computed some batches ago)

In general, LS-based methods work well in terms of convergence but suffers from computational complexity.
