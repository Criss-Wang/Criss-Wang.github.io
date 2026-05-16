---
title: "Variational Inference"
date: 2021/10/14
categories:
  - Blogs
tags:
  - Probability
  - Bayesian Inference
excerpt: "A practical overview of variational inference, approximate posteriors, KL divergence, ELBO, mean-field assumptions, and the connection to VAEs."
layout: post
mathjax: true
toc: true
---

## Introduction

Bayesian inference asks for the posterior distribution:

$$
p(z \mid x) = \frac{p(x \mid z)p(z)}{p(x)}
$$

The problem is that the denominator $p(x)$ is often difficult to compute because it requires integrating over all latent variables.

Variational inference turns inference into optimization. Instead of computing the exact posterior, we choose a simpler family of distributions and find the member that is closest to the true posterior.

## Approximate Posterior

Let $q_\phi(z)$ be an approximate posterior. The goal is:

```text
Find q_phi(z) that is close to p(z | x).
```

The family of possible $q$ distributions might be simple, such as fully factorized Gaussians. This is called a variational family.

The approximation is only as good as the family allows.

## KL Divergence

Closeness is often measured with KL divergence:

$$
KL(q(z) \| p(z \mid x))
$$

Directly minimizing this is hard because it still involves the unknown posterior. Instead, variational inference optimizes the evidence lower bound.

## ELBO

The evidence lower bound (ELBO) is:

$$
ELBO(q) =
\mathbb{E}_{q(z)}[\log p(x, z)]
-
\mathbb{E}_{q(z)}[\log q(z)]
$$

Maximizing the ELBO is equivalent to minimizing the KL divergence from the approximate posterior to the true posterior, up to a constant.

Another common form is:

$$
ELBO =
\mathbb{E}_{q(z)}[\log p(x \mid z)]
-
KL(q(z) \| p(z))
$$

This form shows the tradeoff:

- Fit the observed data.
- Keep the approximate posterior close to the prior.

## Mean-Field Assumption

Mean-field variational inference assumes the approximate posterior factorizes:

$$
q(z) = \prod_i q_i(z_i)
$$

This makes optimization easier, but it ignores posterior dependencies between variables. It is a useful approximation, not a law of nature.

## Coordinate Ascent

In classical variational inference, each factor $q_i$ can be updated while holding the others fixed. This is coordinate ascent variational inference.

Modern deep learning often uses gradient-based optimization instead, especially when neural networks parameterize the variational distribution.

## Connection to VAEs

Variational autoencoders use variational inference with neural networks.

- The encoder defines $q_\phi(z \mid x)$.
- The decoder defines $p_\theta(x \mid z)$.
- Training maximizes an ELBO.

The same idea appears in a deep learning wrapper: approximate a hard posterior with a tractable learned distribution.

## Practical Tradeoffs

Variational inference is usually faster than sampling methods such as MCMC, especially at scale. The tradeoff is approximation bias.

Use it when:

- Exact inference is intractable.
- You need scalable approximate Bayesian inference.
- A tractable posterior approximation is acceptable.
- You can validate whether the approximation is good enough.

Be careful when posterior uncertainty matters deeply. A convenient approximation can understate uncertainty.

## Closing

Variational inference is best understood as optimization-based approximate inference. Choose a family of distributions, optimize the ELBO, and remember that the result is an approximation whose quality depends on both the model and the variational family.
