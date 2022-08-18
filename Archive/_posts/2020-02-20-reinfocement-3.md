---
title: "Reinforcement Learning - Theoretical Foundations: Part III "
date: 2021-03-07
layout: single
author_profile: true
categories:
  - Reinforcement Learning
tag:
  - Control
  - Monte Carlo
excerpt: "RL Continued - Model-free algorithms"
mathjax: "true"
---
## Model - Free Prediction
### Introduction
This is an important chapter that lays the fundamental work for **model-free control** algorithms. In this blog we shall see a few important ideas (MC, TD, online/offline, forward/backward learning) being discussed. While this chapter is not math-intense, it is imperative for us to remember the concepts before moving onto control algorithms.

To begin with, note that this is a **prediction** problem. Hence we are still only going to predict the final $v_{\pi}(s)$ based on a given $\pi$. However, the **Model-free** part suggests that we no longer require an MDP to be explicitly defined/given. This is because we are going to use a sampling strategy. 
- Sampling:
  If a strategy derives certain functions (in this case $v$) directly via episodes of observations, we say that this strategy applies sampling method.

### Monte Carlo (MC) Policy Evaluation
- MC learns from complete episodes (no bootstrapping)
- Monte-Carlo policy evaluation uses *empirical mean return* instead of *expected return*

1. First-step MC
- For each state $s$, only conisder the first time $t$ that $s$ is visited in each episode (update at most 1 time per run)
- Increment counter $N(s) ← N(s) + 1$
- Increment total return $S(s) ← S(s) + G_t$
- Estimate mean return value $V(s) = S(s)/N(s)$
- The estimator $V$ converges to $v_{\pi}$ when number of visits approaches infinity

2. Every-step MC
- For each state $s$, conisder each time $t$ that $s$ is visited in each episode (update at most 1 time per run)
- For instance, $S_1 = s, S_3 = s$, then we update $N(s) \gets N(s) + 2$ and $S(s) \gets S(s) + G_1 + G_3$ where $G_1 = R_1 + \gamma R_2 + \gamma^2 G_3$
- The remaining part are the same as First-step MC

3. Incremental Monte-Carlo Updates
- Based on the idea $\mu_k = \mu_{k-1} + \frac{1}{k}(x_k - \mu_{k-1})$
- Hence we take perspective of each episodes observations $S_t$ instead of states $s$.
- $N(S_t) ← N(S_t) + 1$
- $V(S_t) ← V(S_t) + \frac{1}{N(S_t)}(G_t - V(S_t))$
- In non-stationary problems, it can be useful to track a running mean, i.e. forget old episodes. Hence the formula is tweeked.  $V(S_t) ← V(S_t) +\alpha(G_t - V(S_t))$

    **Limitation**:
      Since MC only updates when the episode terminates (hence 'complete'), it cannot be applied to process which may run infinitely. 

### Temporal Difference (TD) Learning

- A solution to the 'incomplete' episodes problem by applying bootstrapping
- The $G_t$ above is replaced by an estimate $R_{t+1} + \gamma V(S_{t+1})$ (estimated return) based on the Bellman Expectation Equation.
- $R_{t+1} + \gamma V(S_{t+1})$ is called **TD target**
- $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ is called **TD error**
- As now $V(S_t)$ only depends on the next time-step $t+1$ and not the entire episode result $G_t$, we say it is **online** (and hence the other method which updates everything when episode ends is **offline**). 

### Bias Variance trade-off
- Computation using $G_t$ is unbiased, but with high variance due to all rewards $R_{t: t \to \infty}$from transitions $S_t \to S_{t+1}$ and actions selected at random
- Computation using TD target is biased, but with much lower variance, since we we only have one reward $R_t$ to be varied (note that $V(S_{t+1})$ is known and fixed during update of $V(S_t)$
- By the classical trade-off analysis, we see that
	- TD is more sensitive to inital values
	- TD is usually more efficient than MC

### More comparisons between TD and MC

|  | MC | TD |
| -- | -- | -- |
| converges to solution with | minimum mean-squared error | max likelihood Markov model |
| Convergence with function approximation  | Yes | No |
| Exploits Markov property | No | Yes|
| More efficient in | Non-Markov env | Markov env|

### TD($\lambda$)
Now we explore alternative estimators for $G_t$
- **n-step return**: $G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + ... + \gamma^{n-1} R_{t+n} + \gamma^{n} V(S_{t+n})$
- **$\lambda$-return**:  $G_t^{\lambda} = (1-\lambda)\sum\limits_{k=1}^{\infty}\lambda^{k-1}G_t^{(k)}$

Here by utilising $G^{\lambda}_t$, we have TD($\lambda$) as a new evaluation policy. However, notice from the expression above that $G^{\lambda}_t$ is forward-looking, meaning that this would require $n$ transition steps until the episodes end. This faces exactly the same limitation as MC!!!

### Solution: Backward online evaluation
Instead of look for values in future time-steps, we use values from earlier time-steps. This method is called a backward view of TD($\lambda$). In essence, we update online, every step, from incomplete sequences. 

Based on a theorem, the sum of offline updates is identical for forward-view and backward-view TD($\lambda$).

### Eligibility Trace
The key to Backward TD($\lambda$) is eligbility trace, we intuitively derive contributing factors from earlier time steps as follows:
- *Frequency heuristic*: assign credit to most frequent states
- *Recency heuristic*: assign credit to most recent states ($\gamma$ discount rate)

Combing 2 heuristics above, we obtain a rule $E_0(s) = 0$ and $E_t(s) = \gamma\lambda E_{t-1}(s) + \mathbb{1}\{S_{t} = s\}$.
Hence the new update formula is $V(s) ← V(s) +\alpha\delta_tE_t(s)$. Observe now that we need to update $V$ for every state $s$ upon each time-step. 

### Additonal note
- TD(1) is roughly equivalent to every-visit Monte-Carlo
- TD(0) is exactly equivalent to simple TD

## Model Free Control
### Main objective
Instead of estimating the value function, we try to optimize the value function with an unknown MDP.

### Recap of On-Policy vs Off-Policy
1. On-policy
Learn about $\pi$ from experience sampled from previous rounds in $\pi$.
In essence trying to improve $\pi$ by running it **using current agent iteself**.

2. Off-policy
Learn about $\pi$ from experience sampled from previous rounds (or complete run) in $\pi'$.
In essence trying to improve $\pi$ by **Observing another policy** getting run by **another agent** and **deduce several directions** to improve $\pi$ .

### Generalised Policy Iteration With Monte-Carlo Evaluation
In general we can following the policy iteration method introduced in chapter 3 following 2 steps:
- **Policy evaluation** - Monte-Carlo policy evaluation: 
	- We want to evaluate $V = v_{\pi}$. However, with out MDP, we cannot determine $\pi$ easily using a simple **State-Value Function**. So we must resort to a **Action-Value Function**, i.e., $\pi'(s) = \arg\max\limits_{a \in {\cal A}}q_{\pi}(s,a)$. 
	- We further observe that in each iteration, we must run the full policy to obtain $q_{\pi}(s,a)$ for every action/state pair. This is highly inefficient as we do not know how long it takes. Hence instead we do episodic updates using $Q \approx q_{\pi}$. That is, we do **not fully evaluate** that policy, but **sample state-action pair with current policy for $T$ times per episode** and **immediately improve** the policy upon that. This $T$ manually set by us imposes guarantee on  sampling complexity.
	- We call the strategy above *GILE MC control* as it satisfies the **GILE** property.
	- In conclusion, the evaluation phase is as follows:
		- Sample kth episode using $\pi: (S_1, A_1, R_2, ..., S_T) \sim \pi$ 
		- For each state $S_t$ and action $A_t$ in the episode: 
		- $N(S_t, A_t) \leftarrow N(S_t, A_t) + 1$,  
		- $Q(S_t, A_t) \leftarrow Q(S_t, A_t)  + \frac{1}{N(S_t, A_t) } (G_t - Q(S_t, A_t) )$
- Policy improvement: $\epsilon$-Greedy policy improvement
	- We choose actions that greedily maximizes the $Q$. We allow some degree of exploration by making such greedy choice with $1-\epsilon$ probability.
    
    $$\pi(a\|s)=\left\{ \begin{array}{rcl} \frac{\epsilon}{m} + (1- \epsilon) & if & a^{\star} =\arg\max\limits_{a \in {\cal A}}q_{\pi}(s,a) \\ \frac{\epsilon}{m} & & otherwise \end{array} \right \}$$

	- Nothat this $\epsilon$-Greedy policy works as we always have improvement $v_{\pi}(s) \leq v_{\pi'}(s)$ like the proof shown in DP note.

### GILE property
Greedy in the Limit with Infinite Exploration (GLIE) has the following 2 parts:
- All state-action pairs are explored infinitely many times: $\lim\limits_{k \to \infty}N_k(s,a) = \infty$
- The policy converges on a greedy policy, 
    
    $$\lim\limits_{k \to \infty}{\pi}_k(a\|s) = \mathbf{1}[a =\arg\max\limits_{a' \in {\cal A}}Q_{k}(s,a')]$$

For example, $\epsilon$-greedy is GLIE if $\epsilon$ reduces to zero at $\epsilon_k = \frac{1}{k}$.
This property enables the following theorem:
- GLIE Monte-Carlo control converges to the optimal action-value function: $Q(s,a) \to q_*(s,a)$

### Temporal Difference method
1. From MC to TD
Temporal-difference (TD) learning has several advantages over Monte-Carlo (MC) 
- Lower variance 
- Online 
- Incomplete sequences

2, Sarsa update of $Q$
The most basic form is $Q(S, A) ← Q(S, A) + \alpha (R + \gamma Q(S' , A' ) − Q(S, A))$. Now recall from model-free prediction the variations of TD:
- **n-step Sarsa**
	- $q_t^{(n)} = R_{t+1} + \gamma R_{t+2} + ... + \gamma^{(n-1)}R_{t+n} + \gamma^nQ(S_{t+n})$
	- $Q(S, A) ← Q(S, A) + \alpha (q_t^{(n)} − Q(S, A))$
- **Forward View Sarsa($\lambda$)**
	- $q_t^{\lambda} = (1-\lambda)\sum\limits_{k=1}^{\infty}\lambda^{k-1}q_t^{(k)}$
	- $Q(S, A) ← Q(S, A) + \alpha (q_t^{\lambda} − Q(S, A))$
- **Backward View Sarsa($\lambda$)**: we use *eligibility traces* in an online algorithm
	- $E_0(s, a) = 0$
	- $E_t(s, a) = \gamma\lambda E_{t-1}(s) + \mathbf{1}\{S_{t} = s, A_t = a\}$.
	- In each iteration, we upate $Q(s,a)$ for every $(s,a)$ pair.
	- $\delta_t = R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_{t}, A_{t})$
	- $Q(s,a) ← Q(s,a)  +\alpha\delta_tE_t(s,a)$

3. Off-policy learning
In this case, we evaluate target policy $\pi(a\|s)$ to compute $v_{\pi}(s)$ or $q_{\pi}(s,a)$, but the evaluation was based on another (ongoing or completed) policy run $\mu(a|s): (S_1, A_1, R_2, ..., S_T) \sim \mu$

It has the following advantages:
- Learn from observing humans or other agents 
- Re-use experience generated from old policies 
- Learn about optimal policy while following *exploratory* policy 
- Learn about multiple policies while following **one** policy

4. Importance sampling for Off-policy
We note that we can estimate the expectation of a different distribution via:
  
    $$\mathbb{E}_{X \sim P}[f(X)] = \sum Q(X)\frac{P(X)}{Q(X)}f(X) = \mathbb{E}_{X \sim Q}\left[\frac{P(X)}{Q(X)}f(X)\right]$$

One may trie to use returns generated from $\mu$ to evaluate $\pi$ via multiple sampling corrections:

  $$G_t^{\frac{\pi}{\mu}} = G^{\mu}_t \times \frac{\pi(A_t\mid S_t)}{\mu(A_t\mid S_t)}\frac{\pi(A_{t+1}\mid S_{t+1})}{\mu(A_{t+1}\mid S_{t+1})}...\frac{\pi(A_T\mid S_T)}{\mu(A_T\mid S_T)}$$
  
  and then 
  
  $$Q(S, A) ← Q(S, A) + \alpha (G_t^{\frac{\pi}{\mu}} − Q(S, A))$$

However, this multiple chaining may result in:
- Invalud computation when one of the $\mu(a\|s) = 0$ while $\pi(a\|s) \neq 0$
- Dramatically increasing variance

Hence, we consider adopting TD target $R + \gamma Q(S' , A' )$ for importance sampling instead of actual return $G_t$. This removes the multiple chaining as the expression becomes:
- $Q(S_t, A_t) ← Q(S_t, A_t) + \alpha (\frac{\pi(A_{t}\|S_{t})}{\mu(A_{t}\|S_{t})}(R_{t+1} + \gamma Q(S_{t+1}, A_{t+1} ) )− Q(S_t, A_t))$

Unfortunately, in the above expression, we are still sticking to the $\mu$ policy in choosing $A_{t+1}$ when we update our $Q$ for $\pi$. This is not very reasonable, as our policy could potentially have a better choice of action. Importance sampling discounts this fact. Hence we may seek for alternative solution that removes to need to do importance sampling.

### Q-Learning
Q-learning is a method that resolves the above issue. We now consider $Q$ update based on $\pi$: we choose maximizer $A' \sim \pi(\cdot \| S_{t})$. This allows both behaviour and target policies to improve. Note that in this case, $\mu$ is improved via a $\epsilon$-greedy case since $A'$ is chosen randomly with $\epsilon$ probability and $Q \to q_*$ by theorem.
- $Q(S_t, A_t) ← Q(S_t, A_t) + \alpha (R_{t+1} + \max\limits_{a' \in {\cal A}}\gamma Q(S_{t+1}, a' ) − Q(S_t, A_t))$
