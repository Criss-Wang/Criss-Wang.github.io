---
title: "Variational Autoencoder (VAE)"
date: 2021/02/03
categories:
  - Blogs
tags:
  - Deep Learning
  - Generative Models
excerpt: "A practical explanation of variational autoencoders, latent variables, the encoder-decoder structure, reconstruction loss, KL regularization, and sampling."
layout: post
mathjax: true
toc: true
---

## Introduction

A variational autoencoder (VAE) is a generative model that learns a latent representation of data. It combines an encoder, a decoder, and a probabilistic regularization term that shapes the latent space.

VAEs are useful for:

- Generating samples.
- Learning compressed representations.
- Interpolating between examples.
- Modeling uncertainty in latent variables.

## Autoencoder vs VAE

A standard autoencoder maps an input to a latent vector and reconstructs the input:

```text
input -> encoder -> latent vector -> decoder -> reconstruction
```

A VAE maps an input to a distribution over latent variables, usually a Gaussian:

```text
input -> encoder -> mean and variance -> sample z -> decoder -> reconstruction
```

This makes the latent space smoother and easier to sample from.

## Latent Variable View

The VAE assumes data is generated from latent variables:

$$
z \sim p(z)
$$

$$
x \sim p_\theta(x \mid z)
$$

The prior $p(z)$ is often a standard normal distribution. The decoder learns $p_\theta(x \mid z)$.

The encoder approximates the posterior:

$$
q_\phi(z \mid x)
$$

because the true posterior is usually intractable.

## Objective

The VAE optimizes the evidence lower bound (ELBO):

$$
ELBO =
\mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)]
-
KL(q_\phi(z \mid x) \| p(z))
$$

The first term encourages accurate reconstruction. The KL term keeps the learned latent distribution close to the prior.

In practice, the loss is often written as:

```text
loss = reconstruction_loss + KL_regularization
```

## Reparameterization Trick

Sampling from $q_\phi(z \mid x)$ would normally block gradient flow. The reparameterization trick rewrites sampling as:

$$
z = \mu + \sigma \odot \epsilon,\quad \epsilon \sim N(0, I)
$$

Now gradients can flow through $\mu$ and $\sigma$.

## Practical Notes

VAEs can produce smooth latent spaces, but samples may be blurrier than samples from GANs or diffusion models in image tasks.

Common issues:

- Posterior collapse, where the decoder ignores the latent variable.
- Poor reconstruction if the KL term is too strong.
- Latent dimensions that are hard to interpret.

Common fixes:

- KL annealing.
- Beta-VAE objectives.
- Stronger encoder or decoder design.
- Careful latent dimensionality selection.

## Closing

VAEs are a bridge between deep learning and probabilistic modeling. The central idea is simple: learn a latent distribution that can reconstruct data while remaining structured enough to sample from.
