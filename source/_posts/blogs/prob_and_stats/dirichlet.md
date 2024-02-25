---
title: "Stats in ML: Dirichlet Distribution"
excerpt: "A 'popular' distribution not widely known"
date: 2021/07/21
updated: 2022/03/28
categories:
  - Blogs
tags: 
  - Statistics
  - Machine Learning
layout: post
mathjax: true
toc: true
---
## Dirichlet
Before university, I\'ve never heard of Dirichlet distribution. It seemed to me like a loner in the family of statistics when I first saw it. But soon I discovered that this is a huge misunderstanding. Dirichlet is way too important, too usful in the field of machine learning and statistical learning theories to be ignored by any interested scholars. I know deep down in my heart that I must dedicate one whole blog for it to stress its significance. As we walk along the path, we shall see why it is so great (in the past, present and future). Let\'s begin with a general definition of it.

### Formal definition 
The Dirichlet distribution Dir($\alpha$) is a family of continuous multivariate probability distributions parameterized by a vector $\alpha$ of positive reals. It is a *multivariate generalisation* of the Beta distribution. Dirichlet distributions are commonly used as prior distributions (e.g. for *categorical* and *multinomial* distributions) in Bayesian statistics.

### Conjugate Prior and its usage.
In Bayesian probability theory, we have **Posterior $P(\theta \mid X)$**, **Prior $P(\theta)$** and **Likelihood $P(X\|\theta)$** related via the Bayes\' theorem:

  $$P(\theta \mid X) = \frac{P(\theta) \cdot P(X\mid\theta)}{P(X)} $$

If the posterior distribution $P(\theta \mid X)$ and the prior distribution $P(\theta)$ are from the same probability distribution family, then the prior and posterior are called _conjugate distributions_, and the prior is the _conjugate prior_ for the likelihood function $P(X\|\theta)$.

Note that in many algorithm, we want to find the value of $X$ that maximizes the posterior (Maximum a posteriori). If the prior is some weird distribution, we may not get an analytical form for the posterior. Consequently, more complicated optimization strategies like interior point method may need to be applied, which can be computationally expensive. If both the prior and posterior have the same algebraic form, applying bayes rule to find $P(\theta \mid X)$ is much easier.

### Expression
Analogous to multinomial distribution to binomial distribution, Dirichlet is the multinomial version for the beta distribution. Dirichlet distribution is a family of continuous probability distribution for a discrete probability distribution for $k$ categories $\boldsymbol{\theta} = \{\theta_1, \theta_2, \cdots, \theta_k\}$, where $0 \leq \theta_i \leq 1$ for $i \in [1,k]$ and $\sum_{i=1}^k \theta_i = 1$, denoted by $k$ parameters $\boldsymbol{\alpha} = \{\alpha_1, \alpha_2, \cdots, \alpha_k\}$. Formally, we denote $P(\boldsymbol{\theta}\|\boldsymbol{\alpha}) \sim \text{Dir}(\boldsymbol{\alpha})$.

The expression is then $P(\boldsymbol{\theta}\|\boldsymbol{\alpha}) = \frac1{B(\boldsymbol{\alpha})}\prod_{i=1}^k {\theta_i}^{\alpha_i-1}$ where $B(\boldsymbol{\alpha}) = \frac{\prod_{i=1}^k\Gamma(\alpha_i)}{\Gamma(\sum_{i=1}^k\alpha_i)}$ is some constant **normalizer**. 

Think of $P(\boldsymbol{\theta}\|\boldsymbol{\alpha})$ as the probability density associated with **_θ_** which is used for a multinomial distribution, given that our Dirichlet distribution has parameter $\alpha$.

- Moments explicit expressions
  - $E\left(p_i\right) =\frac{\alpha_i}{\sigma}=\alpha_i^{\prime}$
  - $\operatorname{Var}\left(p_i\right) =\frac{\alpha_i(\sigma-\alpha)}{\sigma^{2}(\sigma+1)}=\frac{\alpha_i^{\prime}\left(1-\alpha_i^{\prime}\right)}{(\sigma+1)}$
  - $\operatorname{Cov}\left(p_i, p_j\right) =\frac{-\alpha_i \alpha_j}{\sigma^{2}(\sigma+1)}$

To see that Dirichlet distribution is the conjugate prior for multinomial distribution, consider prior $P(\boldsymbol{\theta}\|\boldsymbol{\alpha}) \sim \text{Dir}(\boldsymbol{\alpha})$, and likelihood $P(\boldsymbol{x}\|\boldsymbol{\theta};n) \sim \text{Mult}(n,\boldsymbol{\theta})$, where $\boldsymbol{x} = \{x_1,x_2,\cdots,x_k\}$ is the sample/result representing $x_i$ success out of $n$ trials for each object $i$. 

$$
\begin{align}
  P(\boldsymbol{\theta}\|\boldsymbol{x};n,\boldsymbol{\alpha}) &\propto P(\boldsymbol{x}\|\boldsymbol{\theta};n) P(\boldsymbol{\theta}\|\boldsymbol{\alpha}) \\\\
  &=\prod_{i=1}^k \binom{n - \sum_{j=1}^{i-1}x_j}{x_i} \prod_{i=1}^k {\theta_i}^{x_i} \frac1{B(\boldsymbol{\alpha})}\prod_{i=1}^k {\theta_i}^{\alpha_i-1} \\\\
  &= \frac{\prod_{i=1}^k \binom{n - \sum_{j=1}^{i-1}x_j}{x_i}}{B(\boldsymbol{\alpha})}\prod_{i=1}^k {\theta_i}^{x_i + \alpha_i-1} \\\\
  &\propto \prod_{i=1}^k {\theta_i}^{x_i + \alpha_i-1}
\end{align}
$$

Therefore, $P(\boldsymbol{\theta}\|\boldsymbol{x};n,\boldsymbol{\alpha}) \sim \text{Dir}(\boldsymbol{x} + \boldsymbol{\alpha})$. We can intrepret this as \"given prior Dirichlet distribution (with param $\boldsymbol{\alpha}$) of probability vector $\boldsymbol{\theta}$ for a total of $k$ objects and an observation vector $\boldsymbol{x}$, the posterior belief of the $\boldsymbol{\theta}$ is a new Dirichlet Distribution with param ($\boldsymbol{\alpha + x}$)\". 

Note the key points here are:

- Distributino has 2 parameters: the scale (or concentration) $\sigma=\sum_i \alpha_i$, and the base measure $\left(\alpha_1^{\prime}, \ldots, \alpha_k^{\prime}\right), \alpha_i^{\prime}=\alpha_i / \sigma$. A Dirichlet with small concentration $\sigma$ favors extreme distributions, but this prior belief is very weak and is easily overwritten by data.

- It shall be seen as a generalization of Beta:
  - Beta is a distribution over binomials (in an interval $p \in[0,1]$ );
  - Dirichlet is a distribution over Multinomials (in the so-called simplex $\sum_i p_i=1 ; p_i \geq 0$ ).

If we want to marginalize the parameters out (often used in ML models for parameter optimization) we can use the following formula:

$$
p\left(x_1, \ldots, x_k \mid \alpha\_1, \ldots, \alpha_k\right)=\frac{\left\\{\prod_i \alpha_i^{x_i}\right\\}}{\sigma^N}
$$

If we want to make prediction via conditional pdf of new data given previous data, we can use the following formula instead:

$$
p\left(\text{new-result}=j \mid x_1, \ldots, x_k, \alpha\_1, \ldots, \alpha_k \right)=\frac{\alpha_j+x_j}{\sigma+N}
$$

### Side note
The above section gives a comprehensive view of dirichlet distribution. However, a more widely applied technique is **Dirichlet Process**. It is similar to *Gaussian Process*, but uses Dirichlet as conjugate prior instead on problems with multinomial likelihood (e.g. Latent Dirichlet Allocation). We\'ve discussed this idea in the topic modeling blog. Interested readers can go that that blog for details.

### References
- [Dirichlet distribution](towardsdatascience.com/dirichlet-distribution-a82ab942a879)
- [Lei Mao\'s Blog](https://leimao.github.io/blog/Introduction-to-Dirichlet-Distribution/)

