---
title: "Gradient Descent Algorithm and Its Variants"
date: 2020/11/13
categories:
  - Blogs
tags:
  - Optimization
  - Machine Learning
excerpt: "A practical explanation of gradient descent, stochastic gradient descent, mini-batch training, momentum, adaptive learning rates, and common optimization issues."
layout: post
mathjax: true
toc: true
---

## Introduction

Gradient descent is the basic optimization idea behind many machine learning algorithms. It updates model parameters in the direction that reduces the loss.

For parameters $\theta$, loss $J(\theta)$, and learning rate $\alpha$:

$$
\theta_{t+1} = \theta_t - \alpha \nabla J(\theta_t)
$$

The gradient points toward the steepest increase in the loss, so subtracting it moves toward lower loss.

## Batch Gradient Descent

Batch gradient descent computes the gradient using the entire training dataset before each update.

Pros:

- Stable gradient estimate.
- Easy to reason about.
- Useful for small datasets and convex problems.

Cons:

- Expensive on large datasets.
- Requires more memory.
- Updates happen slowly when the dataset is large.

## Stochastic Gradient Descent

Stochastic gradient descent (SGD) updates parameters using one example at a time:

$$
\theta_{t+1} = \theta_t - \alpha \nabla J(\theta_t; x_i, y_i)
$$

Pros:

- Fast updates.
- Low memory.
- Noise can help escape shallow local minima.

Cons:

- Noisy training curve.
- Sensitive to learning rate.
- Can bounce around near a minimum.

## Mini-Batch Gradient Descent

Mini-batch gradient descent uses a small batch of examples for each update. This is the standard approach for deep learning.

It balances:

- More stable gradients than pure SGD.
- More efficient hardware use than single-example SGD.
- Lower memory than full-batch training.

The batch size affects optimization, memory, throughput, and generalization. Changing batch size often means retuning the learning rate.

## Learning Rate

The learning rate controls step size.

If it is too small:

- Training is slow.
- The model may appear stuck.

If it is too large:

- Loss may diverge.
- Training may oscillate.
- NaNs can appear.

Learning-rate schedules help:

- Step decay.
- Exponential decay.
- Cosine decay.
- Warmup plus decay.
- Reduce on plateau.

Warmup is especially useful for large neural networks where early gradients can be unstable.

## Momentum

Momentum smooths noisy gradients by accumulating a velocity term:

$$
v_t = \gamma v_{t-1} + \alpha \nabla J(\theta_t)
$$

$$
\theta_{t+1} = \theta_t - v_t
$$

Momentum helps in ravines where gradients oscillate in one direction but consistently move in another. A common value is $\gamma = 0.9$.

## Adaptive Methods

Adaptive optimizers adjust effective learning rates per parameter.

### AdaGrad

AdaGrad accumulates squared gradients. It works well with sparse features, but its effective learning rate can shrink too much.

### RMSProp

RMSProp uses a moving average of squared gradients, reducing AdaGrad's aggressive decay.

### Adam

Adam combines momentum-like first moment estimates with RMSProp-like second moment estimates. It is a common default for neural networks.

### AdamW

AdamW decouples weight decay from the adaptive update. It is widely used for transformer training and fine-tuning.

For optimizer selection, see [Neural Network Applied: Optimizer Selection](/writing/blogs/mlops/nn-optimizers/).

## Common Optimization Problems

### Vanishing or Exploding Gradients

Gradients may become too small or too large in deep networks. Use normalization, residual connections, careful initialization, gradient clipping, or architecture changes.

### Saddle Points

High-dimensional loss surfaces often contain saddle points. Momentum and adaptive methods can help move through flat regions.

### Poor Scaling

Features with very different scales can slow optimization. Standardization often helps classical models and some neural network inputs.

### Bad Loss or Label Issues

Optimization cannot fix broken targets. If loss does not decrease, check data, labels, output shapes, and whether the model can overfit a tiny batch.

## Practical Checklist

When training behaves badly:

1. Verify the data and labels.
2. Overfit a tiny batch.
3. Lower the learning rate.
4. Add warmup.
5. Track gradient norm.
6. Try gradient clipping.
7. Compare SGD, Adam, and AdamW.
8. Check whether batch size changed the effective learning rate.

Gradient descent is simple in equation form, but real training is a system of choices. The learning rate, batch size, optimizer, schedule, and data quality all interact.
