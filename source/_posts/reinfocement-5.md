---
title: "Reinforcement Learning - Theoretical Foundations: Part V "
excerpt: "RL Continued - Policy Gradient"
date: 2021/01/20
updated: 2022/01/31
categories:
  - Blogs
tags: 
  - Reinforcement Learning
layout: post
mathjax: true
toc: true
---
### Policy Gradient
#### 1. General Overview
- Model based RL:
    - **Pros**
  	- \'Easy\' to learn: via procedure similar to supervised learning
  	- Learns everything from the data
	- **Cons**
  	- Objective captures irrelevant info
  	- May focus on irrelevant details
  	- Computing policy (planning) is non-trivial and expensive (offline/static evaluation)

- Value based RL:
	- **Pros**
  	- Closer to True objective
  	- Fairly well understandable - somewhat similar to regression
	- **Cons**
  	- Still not the true objective (similar to model-based)

- Policy based RL:
	- **Pros**
  	- Always try to find the true objective (directly targeting the policy)
	- **Cons**
  	- Ignore some useful learnable knowledge (like policy state/action values) [but can be overcomed by combining with value-function approximation]

#### 2. Model-free Policy based RL
- We directly parametrize the policy via; $\pi_{\theta}(a\mid s) = p(a\mid s, \theta)$
- Advantages:
	- Better convergence properties (gradient method)
	- Effective in high-dimensional or continuous action spaces
	- Can learn stochastic optimal policies (like a scissor-paper-stone game)
	- Sometimes policies are simple while values and models are more complex (large environment but easy policy)
- Disadvantages:
	- Susceptible to local optima (especially with non-linear function approximator)
	- Often obtain knowledge that is specifc and does not always generalize well (high variance)
	- Ignores a lot of information in the data (when used in isolation)

#### 3. Policy Objective function
Here is a list of functions that can be used potentially measure the quality of policy $\pi_{\theta}$. Each is evaluated depending on the things we are concerned about:
- In episodic environment, we can use the **start value**:
	- $J_1(\theta) = v_{\pi_{\theta}}(s_1)$
- In continuous environment, we can use the **average value**:
	- $J_{av}(\theta) = \sum\limits_{s}\mu_{\pi_{\theta}}(s)v_{\pi_{\theta}}(s)$
	- where $\mu_{\pi_{\theta}}(s) = p(S_t = s \mid \pi_{\theta})$ is the probability of being in state $s$ in the long run (long-term proportion)
- Otherwise, we replace the value function $v_{\pi_{\theta}}(s)$ with the reward function so:
	- $J_{av}(\theta) = \sum\limits_{s}\mu_{\pi_{\theta}}(s)\sum\limits_a\pi_{\theta}(s, a)\sum\limits_r p(r \mid s, a) r$

Since the main target is now to optimize $J(\theta)$, we can now simply apply gradient-based method to solve the problem (in this case, **gradient ascent**)

#### 4. Computing the policy gradient analytically
We know that the most important part of any policy function $J(\theta)$ is just the policy expression $\pi(s,a)$. Hence, assuming that $\pi(s,a)$ is differentiable, we find:

  $$\nabla_{\theta}\pi_{\theta}(s,a) = \pi_{\theta}(s,a) \frac{\nabla_{\theta}\pi_{\theta}(s,a) }{\pi_{\theta}(s,a) } = \pi_{\theta}(s,a) \nabla_{\theta} \log \pi_{\theta}(s,a)$$

We then say that the score function (gradient base) of a policy $\pi_{\theta}$ is just $\nabla_{\theta} \log \pi_{\theta}(s,a)$.

We further note that if $J(\theta)$ is an expectaion function dependent on $\pi_{\theta}(s,a)$, i.e., $J(\theta) = \mathbb{E_{\pi_{\theta}}}[f(S,A)]$, we can always apply this gradient base inside the expectation for gradient computation: 

  $$\nabla_{\theta}J(\theta) = \mathbb{E}_{\pi_{\theta}}[f(S,A)\nabla_{\theta} \log \pi_{\theta}(S,A)]$$

- This is called the **score function trick**.
- One useful property is that $\mathbb{E}[b\nabla_{\theta}\log\pi_{\theta}(A_t\|S_t)] =\mathbb{E}[b\nabla_{\theta}\sum\limits_{a}\pi_{\theta}(A_t\|S_t)] = \mathbb{E}[b\nabla_{\theta} 1] =0$ if b does not depend on the action $A_t$. (**expr 1**)

Now consider $\nabla_{\theta}J(\theta)$ formally, we have the following **Policy Gradient Theorem**:
- For any differentiable policy $\pi_{\theta}(s, a)$,  the policy gradient $\nabla_{\theta}J(\theta) = \mathbb{E}\left[\sum\limits_{t = 0} \nabla_{\theta}\log\pi(A_t\mid S_t) q_{\pi}(S_t, A_t)\right]$ where $q_{\pi}(S_t, A_t)$is the long-term value
- *Proof*: Let\'s consider the expected return $\mathbb{E}[G({\cal F})]$ as the objective $J(\theta)$ where ${\cal F} = S_0,A_0, R_1,...,$ is the filtration. note here $G({\cal F})$ is dependent on $\pi_{\theta}(s,a)$ as it affects the filtration. Now:
	- $\nabla_{\theta}J(\theta) =\nabla_{\theta}\mathbb{E}[G({\cal F})] = \mathbb{E}[(\sum\limits_{t=0}R_{t+1})\nabla_{\theta}\log p({\cal F})]$ where $p({\cal F})$ is policy probabilty of this filtration $\cal F$. (Applying the score function trick)

  $$
  \begin{align}
    \nabla_{\theta}\log p({\cal F}) &= \nabla_{\theta}\log\left[ p(S_0)\pi_{\theta}(A_0\|S_0)p(S_1\|S_0,A_0)\pi_{\theta}(A_1\|S_1)...\right] \\\\
    &=\nabla_{\theta}\left[\log p(S_0)+ \log\pi_{\theta}(A_0\|S_0)+\log p(S_1\|S_0,A_0)+\log\pi_{\theta}(A_1\|S_1)...\right] \\\\
    &=\nabla_{\theta}\left[\log\pi_{\theta}(A_0\|S_0)+\log\pi_{\theta}(A_1\|S_1)...\right] \\\\
    & \quad \text{(as $\log p(S_0),\\; \log p(S_1\|S_0,A_0),...$ don\'t depend on $\theta$)}
  \end{align}
  $$

	- So: $\nabla_{\theta}J(\theta) = \mathbb{E}\left[\left(\sum\limits_{t=0}R_{t+1}\right)\left(\sum\limits_{t=0}\nabla_{\theta}\log\pi_{\theta}(A_t\|S_t)\right)\right]$
	- We further notice that $\left(\sum\limits_{t=0}R_{t+1}\right)$ is now a constant for every $(A_t, S_t)$ pair in $\cal F$ for  $\nabla_{\theta}J(\theta)$. However, for any $k$, $\sum_{t=0}^kR_{t+1}$ does not depend on $A_{k+1}, A_{k+2},...$, and by **expr 1** above, we can see that $\nabla_{\theta}J(\theta) = \mathbb{E}\left[\sum\limits_{t=0}\nabla_{\theta}\log\pi_{\theta}(A_t\|S_t)\left(\sum\limits_{i=t}R_{i+1}\right)\right] = \mathbb{E}\left[\sum\limits_{t=0}\nabla_{\theta}\log\pi_{\theta}(A_t\|S_t)q_{\pi}(S_t, A_t)\right]$.

#### 5. Actor-Critic Algorithm

1. Most basic Q Actor-Critic
- policy gradient still has high variance
- We can use a *critic* to estimate the action-value function: $q_{\pi}(S_t, A_t) \approx Q_{w}(S_t, A_t)$
- Actor-critic algorithms maintain *two* sets of parameters 
	- Critic: Updates action-value function parameters $w$ 
	- Actor: Updates policy parameters $\theta$, in direction suggested by critic
- Actor-critic algorithms follow an approximate policy gradient:
	- $\nabla_{\theta}J(\theta) \approx \mathbb{E}\left[\sum\limits_{t=0}\nabla_{\theta}\log\pi_{\theta}(A_t\|S_t)Q_{w}(S_t, A_t)\right]$
	- $\Delta \theta = \alpha \nabla_{\theta}\log\pi_{\theta}(A_t\|S_t)Q_{w}(S_t, A_t)$
- The critic is solving a familiar problem: policy evaluation, which now can be solve using methods via value-based methods.
- However, this approximation of the policy gradient introduces bias. A biased policy gradient may not find the right solution. We can choose value function approximation carefully so that this bias is removed. This is possible because of the **Compatible Function Approximation Theorem** below:
	- If the following two conditions are satisfied: 
		1. Value function approximator is compatible to the policy $\nabla_wQ_w(s,a) = \nabla_{\theta}\log\pi_{\theta}(s,a)$
		2. Value function parameters w minimise the mean-squared error $\varepsilon = \mathbb{E_{\pi_{\theta}}}[(q_{\pi_{\theta}}(s,a) - Q_w(s,a))^2]$
	- Then $\nabla_{\theta}J(\theta) = \mathbb{E}\left[\sum\limits_{t=0}\nabla_{\theta}\log\pi_{\theta}(A_t\|S_t)Q_{w}(S_t, A_t)\right]$ exactly.

2. Advantage Actor-Critic
- Recall **expr 1**, we can again apply this on $Q_{w}(S_t, A_t)$ to further reduce the variance introduced by a large $Q_{w}(S_t, A_t)$ value. 
- Consider $A_{\pi_{\theta}}(s,a) = Q_{\pi_{\theta}}(s,a) - v_{\pi_{\theta}}(s)$. This is called an **advantage function.** Since $v_{\pi_{\theta}}(s)$ does not depend on the actions, so$\pi_{\theta}(s,a)$ plays no role here. Then **expr 1** with $\nabla_{\theta}J(\theta)$ formula results in the following 
- $\nabla_{\theta}J(\theta) = \mathbb{E}\left[\sum\limits_{t=0}\nabla_{\theta}\log\pi_{\theta}(A_t\|S_t)A_{\pi_{\theta}}(S_t, A_t)\right]$

3. TD Actor-Critic
- We now apply approximation again (Compatible Function Approximation Theorem) on $A_{\pi_{\theta}}(s)$ using the an estimated TD error $\delta_{\pi_{\theta}}(S_t) = R_{t+1} + \gamma v_{\pi_{\theta}}(S_{t+1})- v_{\pi_{\theta}}(S_t)$
- In practice we can use an approximate TD error $\delta_{t} = R_{t+1} + \gamma v_{w}(S_{t+1})- v_{w}(S_t)$
- So now the critic update is $\Delta \theta = \alpha \delta_t \nabla_{\theta}\log\pi_{\theta}(A_t\|S_t)$
- Note given this variant that Critic can estimate value function $v_{\pi_{\theta}}(s)$ from many targets at different time-scales. (MC, TD(0), Forward/Backward TD($\lambda$)

4. Natural Actor-Critic
- refers to [Policy Gradient](https://www.davidsilver.uk/wp-content/uploads/2020/03/pg.pdf) for more on this content.
