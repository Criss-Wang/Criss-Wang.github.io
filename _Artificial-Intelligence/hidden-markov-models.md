---
title: "An overview of Hidden markov model and its algorithms"
date: 2021-04-22
layout: single
author_profile: true
categories:
  - Statistics
tags: 
  - Hidden Markov Models
  - Expectation Maximization
  - Dynamic Programming
excerpt: "Learning stochastic process with more details"
mathjax: "true"
---
## Overview
Stochastic Process is a critical piece of knowledge in statistical learning. It's also an important piece in the increasing popular reinforcement learning field. I feel like I might apply its algorithms or do research work on it in the future. Hence, I create this blog to introduce an important concept, hidden markov model (HMM), and some useful algorithms and their intuitions for HMM.

## 1. Markov Chain

The HMM is based on augmenting the Markov Chain(MC), which is a type of a random process about a set of states. The transition from one state to another depends on certain probabilities $P(X_t = v_i \| X_{t-1} = v_j,\ldots ,X_{1} = v_k), which we define as **transition probability**. The nice property of an MC is that the transition probability of $X_t$ only depends on the previous state $X_{t-1}$, i.e. $P(X_t = v_i \| X_{t-1} = v_j)$. We call it the **Markov Assumption**. This property allows us to produce an each transition graph like this[^1]

<figure style="width: 600px" class="align-center">
    <img src="/images/AI/HMM_1.png" alt="">
    <figcaption></figcaption>
</figure> 
[^1]:https://web.stanford.edu/~jurafsky/slp3/A.pdf

## 2. HMM Definition
A Markov chain is useful when we need to compute a probability for a sequence of observable events. In many cases, however, the events we are interested in are hidden: we don’t observe them directly. For example we don’t normally observe part-of-speech tags in a text. Rather, we see words, and must infer the tags from the word sequence. We call the tags hidden because they are not observed. A hidden Markov model (HMM) allows us to talk about both observed events (like words that we see in the input) and hidden events (like part-of-speech tags) that we think of as causal factors in our probabilistic model. In HMM, we use **observations** $X_t$ to describe observed events with values denoted by $o_t$, and **states** $H_t$ to describe hidden events with values denoted by $q_t$. Note that **Markov Assumption** still holds. In addition, we have output independence $P\left(o_{i} \mid q_{1} \ldots q_{i}, \ldots, q_{T}, o_{1}, \ldots, o_{i}, \ldots, o_{T}\right)=P\left(o_{i} \mid q_{i}\right)$. In HMM, we try to solve the following problems:
- Problem 1 (Likelihood): Given an HMM with transition probabilities $\{P_{ij} = P(H_t = q_j \| H_{t-1} = q_i)\}$ and observation probabilities ${P^H_{ik}=P(X_t = o_i \| H_t = q_k)\}$ and an observation sequence $O$, determine the likelihood $P(O \mid P_{ij}, P^H_{ik})$.
- Problem 2 (Decoding): Given an observation sequence $O$ and an HMM $(P_{ij}, P^H_{ik})$, discover the best hidden state sequence $Q$. 
- Problem 3 (Learning): Given an observation sequence $O$ and the set of states in the HMM, learn the HMM parameters $P_{ij}$ and $P^H_{ik}$.

## Algorithms
###  The Forward Algorithm - Likelihood solver
The forward algorithm is a dynamic programming method that computes the observation probability by summing over the probabilities of all possible hidden state paths that could generate the observation sequence, but it does so efficiently by implicitly folding each of these paths into a single forward trellis, which computes the probability of being in state $q_j$ after seeing the first $t$ observations, given the parameteres $(P_{ij}, P^H_{ik})$, i.e.

$$
\alpha_{t}(j)=P\left(o_{1}, o_{2} \ldots o_{t}, q_{t}=j \mid P_{ij}, P^H_{ik} \right)=\sum_{i=1}^{N} \alpha_{t-1}(i) P_{i j} P^{H}_{j}\left(o_{t}\right) \tag{1}
$$

From above, we can quickly derive the result by:
1. first initialize $\alpha_{1}(j)=\pi_{j} P^H_{j}\left(o_{1}\right), \quad 1 \leq j \leq N$.
2. Recrusively apply the above expression (1) for $1 \leq j \leq N, 1<t \leq T$.
3. Compute $P(O \mid P_{ij}, P^H_{ik})=\sum_{i=1}^{N} \alpha_{T}(i)$

### The Viterbi Algorithm - Decoding solver
Like forwarding algorithm, Viterbi is also DP that makes uses of a dynamic programming Viterbi trellis. The idea is to process the observation sequence left to right, filling out the trellis. Each cell of the trellis, $v_{t}(j)$, represents the probability that the HMM is in state $j$ after seeing the first $t$ observations and passing through the most probable state sequence $q_{1}, \ldots, q_{t-1}$, given the parameters $(P_{ij}, P^H_{ik})$. The value of each cell $v_{t}(j)$ is computed by recursively taking the most probable path that could lead us to this cell. 

$$
v_{t}(j)=\max_{q_{1}, \ldots, q_{t-1}} P\left(q_{1} \ldots q_{t-1}, o_{1}, o_{2} \ldots o_{t}, q_{t}=j \mid (P_{ij}, P^H_{ik})\right) =\max _{i=1,\ldots, N} v_{t-1}(i) P_{i j}P^H_{j}\left(o_{t}\right)\tag{2}
$$

Here, a major difference from (1) is that we now take the most probable of the extensions of the paths that lead to the current cell. In addition to the max value $v_{t}(j)$, we shall also keep track of the solution 

$$bp_t(j) = \underset{i=1,\ldots,N}{\operatorname{argmax}}v_{t-1}(i) P_{i j}P^H_{j}\left(o_{t}\right) \qquad (bp_1(j) = 0)$$

This is called backpointers. We need this value because while the forward algorithm needs to produce an observation likelihood, the Viterbi algorithm must produce a probability and also the most likely state sequence. We compute this best state sequence by keeping track of the path of hidden states that led to each state, and Viterbi backtrace then at the end backtracing the best path to the beginning (the Viterbi backtrace).

Finally, we can compute the optimal score and path 

$$
V^* = \max _{i=1,\ldots, N}v_{T}(i) \\
P^* = \underset{i=1,\ldots,N}{\operatorname{argmax}} v_{T}(i)
$$

We use a demo code to illustrate the process. The code for *forwarding* algorithm and *Forward-Backward* Algorithm can be implmented in a similar fashion.
```python
import numpy as np
import pandas as pd

# Define the problem setup
obs_map = {'Cold':0, 'Hot':1}
obs = np.array([1,1,0,1,0,0,1,0,1,1,0,0,0,1])

inv_obs_map = dict((v,k) for k, v in obs_map.items())
obs_seq = [inv_obs_map[v] for v in list(obs)]

print("Simulated Observations:\n",pd.DataFrame(np.column_stack([obs, obs_seq]),columns=['Obs_code', 'Obs_seq']) )

pi = [0.6,0.4] # initial probabilities vector
states = ['Cold', 'Hot']
hidden_states = ['Snow', 'Rain', 'Sunshine']
pi = [0, 0.2, 0.8]
state_space = pd.Series(pi, index=hidden_states, name='states')
a_df = pd.DataFrame(columns=hidden_states, index=hidden_states)
a_df.loc[hidden_states[0]] = [0.3, 0.3, 0.4]
a_df.loc[hidden_states[1]] = [0.1, 0.45, 0.45]
a_df.loc[hidden_states[2]] = [0.2, 0.3, 0.5]
print("\n HMM matrix:\n", a_df)
a = a_df.values

observable_states = states
b_df = pd.DataFrame(columns=observable_states, index=hidden_states)
b_df.loc[hidden_states[0]] = [1,0]
b_df.loc[hidden_states[1]] = [0.8,0.2]
b_df.loc[hidden_states[2]] = [0.3,0.7]
print("\n Observable layer  matrix:\n",b_df)
b = b_df.values

# Apply the Viterbi Algorithm based on http://www.blackarbs.com/blog/introduction-hidden-markov-models-python-networkx-sklearn/2/9/2017
def viterbi(pi, a, b, obs):
    
    nStates = np.shape(b)[0]
    T = np.shape(obs)[0]
    
    # init blank path
    path = path = np.zeros(T,dtype=int)
    # delta --> highest probability of any path that reaches state i
    delta = np.zeros((nStates, T))
    # phi --> argmax by time step for each state
    phi = np.zeros((nStates, T))
    
    # init delta and phi 
    delta[:, 0] = pi * b[:, obs[0]]
    phi[:, 0] = 0

    print('\nStart Walk Forward\n')    
    # the forward algorithm extension
    for t in range(1, T):
        for s in range(nStates):
            delta[s, t] = np.max(delta[:, t-1] * a[:, s]) * b[s, obs[t]] 
            phi[s, t] = np.argmax(delta[:, t-1] * a[:, s])
            print('s={s} and t={t}: phi[{s}, {t}] = {phi}'.format(s=s, t=t, phi=phi[s, t]))
    
    # find optimal path
    print('-'*50)
    print('Start Backtrace\n')
    path[T-1] = np.argmax(delta[:, T-1])
    for t in range(T-2, -1, -1):
        path[t] = phi[path[t+1], [t+1]]
        print('path[{}] = {}'.format(t, path[t]))
        
    return path, delta, phi

# Run the algo
path, delta, phi = viterbi(pi, a, b, obs)
state_map = {0:'Snow', 1:'Rain', 2:'Sunshine'}
state_path = [state_map[v] for v in path]
pd.DataFrame().assign(Observation=obs_seq).assign(Best_Path=state_path)
```

### Forward-backward algorithm - Learning solver
The standard algorithm for HMM training is the forward-backward, or Baum-Welch Welch algorithm, a special case of the *Expectation-Maximization* (EM) algorithm. The algorithm will let us train both the transition probabilities $P_{ij}$ and the emission probabilities $P^H_{ik}$ of the HMM. EM is an iterative algorithm, computing an initial estimate for the probabilities, then using those estimates to computing a better estimate, and so on, iteratively improving the probabilities that it learns.

For a real HMM, we only get observations, and cannot compute which observation is from which state directly from an observation sequence since we don't know which path of states was taken through the machine for a given input. What's more, we don't even know when is a hidden state present. The Baum-Welch algorithm solves this by iteratively estimating the number of times a state occurs. We will start with an estimate for the transition and observation probabilities and then use these estimated probabilities to derive better and better probabilities. And we're going to do this by computing the forward probability for an observation and then dividing that probability mass among all the different paths that contributed to this forward probability.

First, we define backward probability $\beta$, the probability of seeing the observations from time $t+1$ to the end, given that we are in state $i$ at time $t$ (and given the automaton $\lambda$ ):

$$
\beta_{t}(i)=P\left(o_{t+1}, o_{t+2} \ldots o_{T} \mid q_{t}=i, \lambda\right)
$$

We can actually think of it as a reverse of the forwarding probability $\alpha_{t}(j)$, and the computation is just the ''reverse'' of that for $\alpha_{t}(j)$, so now the computation of $P(O\|P_{ij}, P^H_{ik})$:
1. First initialize $\beta_{T}(i)=1, \quad 1 \leq i \leq N$.
2. Recrusively apply $\beta_{t}(i)=\sum_{j=1}^{N} P_{i j} P^H_{j}\left(o_{t+1}\right) \beta_{t+1}(j), \quad 1 \leq i \leq N, 1 \leq t<T$.
3. Compute $P(O \mid \lambda)=\sum_{j=1}^{N} \pi_{j} P^H_{j}\left(o_{1}\right) \beta_{1}(j)$

Now, we start the actual work. To estimate the $P_{ij}, P^H_{ik}$, we may adopt a simple maximum likelihood estimation

$$
\widehat{P}_{i j}=\frac{\text { expected number of transitions from state } i \text { to state } j}{\text { expected number of transitions from state } i} \\
\widehat{P^H}_{j}\left(v_{k}\right)=\frac{\text { expected number of times in state } j \text { and observing symbol } v_{k}}{\text { expected number of times in state } j}
$$

How do we compute the expected number of transitions from state $i$ to state $j$? Here's the intuition. Assume we had some estimate of the probability that a given transition $i \rightarrow j$ was taken at a particular point in time $t$ in the observation sequence. If we knew this probability for each particular time $t$, we could sum over all times $t$ to estimate the total count for the transition $i \rightarrow j$. We can then compute probability $\xi_{t}$ of being in state $i$ at time $t$ and state $j$ at time $t+1$, given the observation sequence and of course the model via bayes' rule:

$$
\xi_{t}(i, j)=P\left(q_{t}=i, q_{t+1}=j \mid O, P_{ij}, P^H_{ik}\right) = \frac{P\left(q_{t}=i, q_{t+1}=j, O \mid P_{ij}, P^H_{ik}  \right))}{P(O \mid P_{ij}, P^H_{ik} )}
$$

since we can easily compute 

$$
P\left(q_{t}=i, q_{t+1}=j, O \mid P_{ij}, P^H_{ik}  \right))=\alpha_{t}(i) P_{i j} P^H_{j}\left(o_{t+1}\right) \beta_{t+1}(j); \\
P(O \mid P_{ij}, P^H_{ik} )=\sum_{j=1}^{N} \alpha_{t}(j) \beta_{t}(j);
$$

we get

$$
\widehat{P}_{i j}=\frac{\sum_{t=1}^{T-1} \xi_{t}(i, j)}{\sum_{t=1}^{T-1} \sum_{k=1}^{N} \xi_{t}(i, k)} \tag{3}.
$$

Similarly, by considering expected number of times in state $j$ and observing symbol $v_k$. Define the probability of being in state $j$ at time $t$ as $\gamma_{t}(j)$, compute

$$
\gamma_{t}(j)=\frac{P\left(q_{t}=j, O \mid P_{ij}, P^H_{ik}\right)}{P(O \mid P_{ij}, P^H_{ik})} = \frac{\alpha_{t}(j) \beta_{t}(j)}{P(O \mid P_{ij}, P^H_{ik})}
$$

We are ready to compute $b$. For the numerator, we sum $\gamma_{t}(j)$ for all time steps $t$ in which the observation $o_{t}$ is the symbol $v_{k}$ that we are interested in. For the denominator, we sum $\gamma_{t}(j)$ over all time steps $t$. The result is the percentage of the times that we were in state $j$ and saw symbol $v_{k}$ (the notation $\sum_{t=1 \text { s.t. } O_{t}=v_{k}}^{T}$ means "sum over all $t$ for which the observation at time $t$ was $\left.v_{k} "\right)$

$$
\widehat{P^H}_{j}\left(v_{k}\right)=\frac{\sum_{t=1\text { s.t. $O_{t}=v_{k}$} }^{T} \gamma_{t}(j) }{\sum_{t=1}^{T} \gamma_{t}(j)} \tag{4}.
$$

Using (3), (4), we can apply EM algorithm easily, as demonstrated below:
- Initialize $P_{ij}, P^H_{ik}$
- iterate until Convergence:
  - E-step:

      $$
      \gamma_{t}(j) = \frac{\alpha_{t}(j) \beta_{t}(j)}{\sum_{j=1}^{N} \alpha_{t}(j) \beta_{t}(j)}\forall t \text{and} j \\
      \xi_{t}(i, j) = \frac{\alpha_{t}(i) \widehat{P}_{i j} \widehat{P^H}_{j}\left(o_{t+1}\right) \beta_{t+1}(j)}{\sum_{j=1}^{N} \alpha_{t}(j) \beta_{t}(j)} \forall t, i \text{and} j
      $$

  - M-step:

      $$
      \widehat{P}_{i j}=\frac{\sum_{t=1}^{T-1} \xi_{t}(i, j)}{\sum_{t=1}^{T-1} \sum_{k=1}^{N} \xi_{t}(i, k)} \\
      \widehat{P^H}_{j}\left(v_{k}\right)=\frac{\sum_{t=1\text { s.t. $O_{t}=v_{k}$} }^{T} \gamma_{t}(j) }{\sum_{t=1}^{T} \gamma_{t}(j)} 
      $$

### This concludes the blog, thank you!