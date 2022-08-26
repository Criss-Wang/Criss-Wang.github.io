---
title: "Variational Autoencoder (VAE)"
excerpt: "Combine variational inference with representation learning"
date: 2021/02/03
updated: 2022/03/17
categories:
  - Blogs
tags: 
  - Statistical Inference
  - Representaiton Learning
layout: post
mathjax: true
toc: true
---
### A short Intro to VAE

#### 1. Background
There mainly 2 types of deep generative models:
- Generative Adversarial Network (GAN)
- Variational Autoencoder (VAE)

We will discuss about VAE in this blog. In future blogs, we will venture into the details of GAN.

#### 2. A basic intuition
A VAE is an autoencoder whose encodings distribution is regularised (via variational inferenece) during the training in order to ensure that its latent space has good properties allowing us to generate some new data.

#### 3. Encoder & Decoder
**encoder** is an agent that transform older featurer representation to a new set of feature representation (usually a lower dimension) using selection or extraction and **decoder** is an agent producing the reverse process. The encoded representation can span a feature space of certain dimensionality. We call it **latent space**.  Furthermore,  we know that certain properties/information of original features may be lost if we encode them. So we categorize transformations as **lossy** or **lossless** transformation. we are looking for the pair that keeps the maximum of information when encoding and, so, has the minimum of reconstruction error when decoding. 

#### 4. Autoencoder
Autoencoder is done by **setting an encoder and a decoder as neural networks** and to **learn the best encoding-decoding scheme using an iterative optimisation process**. So, at each iteration we feed the autoencoder architecture (the encoder followed by the decoder) with some data, we compare the encoded-decoded output with the initial data and backpropagate the error through the architecture to update the weights of the networks. We usually use a Mean Square Error as the loss function for backpropagation. This is often compared with **PCA**. When the structure of encoder/decoder gets deeper and more non-linear, we observe that autoencoder can still proceed to a high dimensionality reduction while keeping reconstruction loss low. 

There are mainly 2 drawbacks of Autoencoder:
- the lack of interpretable and exploitable structures in the latent space (**lack of regularity**)
- Difficulty in reducing a large number of dimensions **while keeping the major part of the data structure information in the reduced representations**

As a result of the above two drawbacks, we may generate meaningless data if we simply encode into a latent space and sample random points from it to decode (a generative model often requires this). This issue leads to the need of regularization for the latent\'s space distribution. This is the motivation for **Variational Autoencoder**.

### VAE in detail
The key idea for VAE is that, **instead of encoding an input as a single point, we encode it as a distribution over the latent space**. In essence, we don\'t have point-wise estimation, but an estimation of original inputs\' distribution (hence Bayesian inference is here of significant help). Next the evalution of an error is based on a new sample **drawn** from the distribution estimator and compared with original sample.

(In practice, the encoded distributions are chosen to be normal so that the encoder can be trained to return the mean and the covariance matrix that describe these Gaussians. See why this is the case in mathematical details below)

Because of regularization, we now have an additional term inside loss, which is the **KL divergence**. We see that the KL divergence between 2 Gaussians has a closed form and hence can be computed easily. 

#### 1. All the math down here
We require two assumptions:
1. a latent representation $z$ is sampled from the prior distribution $p(z)$;
2. the data $x$ is sampled from the conditional likelihood distribution $p(x\|z)$

Now we note here that the “probabilistic decoder” is naturally defined by $p(x\|z)$, that describes the **distribution of the decoded variable given the encoded one**, whereas the “probabilistic encoder” is defined by $p(z\|x)$, that describes the **distribution of the encoded variable given the decoded one**. 

These two expressions remind us easily of the Bayes Rule, which we have $p(z\|x) =\frac{p(x\|z)p(z)}{p(x)}$. We assume $p(x) \sim {\cal N}(0, I)$ and $p(x\|z) \sim {\cal N}(f(z), cI), \quad f\in F, c > 0$. Since f is arbitrary, the evidence term $f(x)$ is often an intractable integral. Hence we need Variational Inference (**VI**) to help us approximate $p(z\|x)$ directly via a easily tractable distribution (often a Gaussian distribution $q_x(z)$). 

We define $q_x(z) \sim {\cal N}(g(x), h(x))$ where $g(x), h(x)$ are parametrized functions. With uncertainty in $f\in F$, we aim to obtain optimal $(f^{\star}, g^{\star}, h^{\star})$ as 

  $${\operatorname{argmax}}\_{(f,g,h)\in  F\times G\times H} \left(\mathbb{E}\_{z\sim q_x}\left(-\frac{\Vert x-f(z)\Vert^2}{2c}\right) - KL(q_x(z)\Vert p(z))\right) \tag{1}$$ 


#### 2. Practical idea: Neural Network
Now that we have an optimization problem which may be solved using NN, we still need to address a few issues. First, the entire space of $F\times G\times H$ is too large, so we need to constrain the optimisation domain and decide to express f, g and h as neural networks. In practice, g and h are not defined by two completely independent networks but share a part of their architecture and their weights (with dependent paramters) 

For simplicity of computation, we often require $q_x(z) \sim {\cal N}(g(x), h(x))$ to be a multidimensional Gaussian distribution with diagonal covariance matrix. With this assumption, h(x) is simply the vector of the diagonal elements of the covariance matrix and has then the same size as g(x). On the other hand NN models $f$, which represents the mean of $p(x\|z)$ assumed as a Gaussian with fixed covariance.

Using these ideas, we can first sample $z \sim q_x(z)$ and evaluate an $f(z)$ followed by error evaluation and backpropagation. Note here that the sampling process has to be expressed in a way that allows the error to be backpropagated through the network. A simple trick, called **reparametrisation trick**, is used to make the gradient descent possible despite the random sampling that occurs halfway of the architecture. This is done by producing a concrete sample from $q_x(z): z = \zeta h(x) + g(x), \quad \zeta \sim {\cal N}(0,I)$. So we have a **Monte-carlo** estimation via sampling to replace the expectation term in the loss term **(1)**. We then backpropagate error to obtain the final result. 

### Code implementation
**To Be Updated**

