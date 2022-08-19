---
layout: single
title: "Gradient Descent Algorithm and Its Variants!"
date: 2020/11/13
updated: 2022/8/19
categories:
  - Neural Network
tags: 
  - Gradient Descent
excerpt: "The most important technique in neural networks (as of now)"
---
## Overview of Gradient Descent
**Optimization** refers to the task of minimizing/maximizing an objective function \\(f(x)\\) parameterized by $x$. In machine/deep learning terminology, it's the task of minimizing the cost/loss function $J(w)$ parameterized by the model's parameters \\(w \in \mathbb{R}^d\\). Optimization algorithms (in case of minimization) have one of the following goals:
- Find the global minimum of the objective function. This is feasible if the objective function is convex, i.e. any local minimum is a global minimum.
- Find the lowest possible value of the objective function within its neighbor. That's usually the case if the objective function is not convex as the case in most deep learning problems.

There are three kinds of optimization algorithms:

- Optimization algorithm that is not iterative and simply solves for one point.
- Optimization algorithm that is iterative in nature and converges to acceptable solution regardless of the parameters initialization such as gradient descent applied to logistic regression.
- Optimization algorithm that is iterative in nature and applied to a set of problems that have non-convex cost functions such as neural networks. Therefore, parameters' initialization plays a critical role in speeding up convergence and achieving lower error rates.

**Gradient Descent** is the most common optimization algorithm in *machine learning* and *deep learning*. It is a first-order optimization algorithm. This means it only takes into account the first derivative when performing the updates on the parameters. On each iteration, we update the parameters in the opposite direction of the gradient of the objective function $J(w)$ w.r.t to the parameters where the gradient gives the direction of the steepest ascent. The size of the step we take on each iteration to reach the local minimum is determined by the learning rate $\alpha$. Therefore, we follow the direction of the slope downhill until we reach a local minimum. 

**Gradient Descent** is the most common optimization algorithm in *machine learning* and *deep learning*. It is a first-order optimization algorithm. This means it only takes into account the first derivative when performing the updates on the parameters. On each iteration, we update the parameters in the opposite direction of the gradient of the objective function $J(w)$ w.r.t to the parameters where the gradient gives the direction of the steepest ascent. The size of the step we take on each iteration to reach the local minimum is determined by the learning rate $\alpha$. Therefore, we follow the direction of the slope downhill until we reach a local minimum. 

In this notebook, we'll cover gradient descent algorithm and its variants: *Batch Gradient Descent, Mini-batch Gradient Descent, and Stochastic Gradient Descent*.

Let's first see how gradient descent and its associated steps works on logistic regression before going into the details of its variants. For the sake of simplicity, let's assume that the logistic regression model has only two parameters: weight $w$ and bias $b$.

1. Initialize weight $w$ and bias $b$ to any random numbers.
2. Pick a value for the learning rate $\alpha$. The learning rate determines how big the step would be on each iteration.
    * If $\alpha$ is very small, it would take long time to converge and become computationally expensive.
    * IF $\alpha$ is large, it may fail to converge and overshoot the minimum.
        
      Therefore, plot the cost function against different values of $\alpha$ and pick the value of $\alpha$ that is right before the first value that didn't converge so that we would have a very fast learning algorithm that converges(Figure 1).

      <figure style="width: 300px" class="align-center">
        <img src="/images/Machine%20learning/learning_rate.png" alt="">
        <figcaption>Figure 1</figcaption>
      </figure> 
    * The most commonly used rates are : *0.001, 0.003, 0.01, 0.03, 0.1, 0.3*.