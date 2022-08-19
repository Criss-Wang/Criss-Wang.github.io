---
title: "Reinforcement Learning - Theoretical Foundations: Part II "
date: 2021-03-06
layout: single
author_profile: true
categories:
  - Reinforcement Learning
tag:
  - Dynamic Programming
excerpt: "RL Continued - Dynamic Programming"
mathjax: "true"
---
## Dynamic Programming in RL
### Introduction
- DP assumes full knowledge of the MDP
- __A prediction problem__: The input is an MDP/MRP and a policy $\pi$. The output is a value function $v_{\pi}$.
- __A control problem__: The input is an MDP. The output is the optimal value function $v_*$ and an optimal policy $\pi_{\star}$. 

### Synchronous DP
The following table summarizes the type of problems that is solved synchronously via iteration/evaluation algorithms:

| Problem       | Bellman Equation | Algorithm| 
| ------------- | -------------    | -----    |
| Prediction      | Bellman Expectation Equation| Iterative Policy Evaluation |
| Control     | Bellman Expectation Equation Policy Iteration + Greedy Policy Improvement    |  Policy Iteration |
| Control | Bellman Optimality Equation |    Value Iteration |

### Iterative Policy Evaluation
- Problem: evaluate a given policy $\pi$
- __Algo sketch__:
	1. Assign each state with an initial value (for example: $v_0(s) = 0 \;\;\forall s\in S$)
	2. Following the policy, compute the updated value function $v_i(s)$ using the Bellman Expectation Equation **$v^{k+1} = {\cal R}^{\pi} + \gamma {\cal P}^{\pi}v^k$**
	3. Iterate until convergence (proven later)

### Policy Improvement
- Upon Evaluation of a policy $\pi$, we can seek to greedily improve the policy such that we obtain $v_{\pi'}(s) \geq v_{\pi}(s)$. (**expr 1**)
- The greedy approach acts as selecting $\pi '(s) = \arg\max\limits_{a \in {\cal A}}q_{\pi}(s,a)$. (**eq 1**)
- We can prove that __eq 1__ leads to __expr 1__ as follows:
	- In one step: $q_{\pi}(s,\pi '(s)) = \max\limits_{a \in {\cal A}}q_{\pi}(s,a) \geq q_{\pi}(s,\pi (s)) = v_{\pi}(s)$.
	- Note that $\pi'$ is a deterministic policy. Observe that 
	 
	  $$q_{\pi}(s,\pi '(s)) = \mathbb{E}_{\pi'}[R_{t+1} + \gamma v_{\pi}(S_{t+1}) \| S_t = s, A_t = \pi'(s)] = \mathbb{E}_{\pi'}[R_{t+1} + \gamma v_{\pi}(S_{t+1}) \| S_t = s]$$

	- Hence 
	  
    $$
    \begin{align}
      v_{\pi}(s) \leq q_{\pi}(s,\pi '(s)) &= \mathbb{E}_{\pi'}[R_{t+1} + \gamma v_{\pi}(S_{t+1}) \| S_t = s] \\
      &\leq \mathbb{E}_{\pi'}[R_{t+1} + \gamma q_{\pi}(S_{t+1},\pi '(S_{t+1})) \| S_t = s] \\
      &\leq \mathbb{E}_{\pi'}[R_{t+1} + \gamma R_{t+2}  + \gamma^2 q_{\pi}(S_{t+2},\pi '(S_{t+2})) \| S_t = s] \\
      &\leq \mathbb{E}_{\pi'}[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ...\| S_t = s] = v_{\pi'}(s) \\
    \end{align}
    $$

- Basically, we find that this method is equivalent to solving the Bellman Optimality equation. So we obtain $\pi$ as an optimal policy
- Note that this process of policy iteration always converges to $\pi_*$. 
- __Drawback__: Policy Iteration always Evaluation an entire Policy before it starts to improve on the policy. This may be highly inefficient if the evaluation of a policy takes very long time (e.g. infinite MDP)
- To deal with the __Drawback__, we utilise __DP__ -> Value Iteration.

### Value Iteration
- We improve the value function $v_i(s)$ in each iteration
- Note that we are __only__ improving the value function, where this value function is based on any explicit policy
- Intuition: start with final rewards (again all 0 for example) and work backwards
- Now assume we know the solution to a subproblem $v_{\star}(s')$, then we can find $v_{\star}(s)$ by one-step look ahead:
	- $v_{\star}(s) \gets\max\limits_{a \in {\cal A}}{\cal R}^a_s + \gamma \sum\limits_{s' \in S} {\cal P}^a_{ss'}v_{\star}(s')$
- Therefore, we can always update the value function in each iteration backwards until convergence.

### Contraction Mapping Theorem
- To be updated upon publishing the markdown 
- Refer to page 28 - 42 [(DP)](https://www.davidsilver.uk/wp-content/uploads/2020/03/DP.pdf)

### Asynchronous DP
There are 3 simple ideas, which I haven't learning in detail:
- In-place dynamic programming 
- Prioritised sweeping 
- Real-time dynamic programming 
