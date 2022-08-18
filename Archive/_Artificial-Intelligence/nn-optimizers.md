---
title: "Neural Network Applied: Optimizer Selection"
date: 2020-11-30
layout: single
author_profile: true
categories:
  - Neural Network
tags: 
  - Optimization
  - Linear Algebra
excerpt: "Choose a good optimization strategy is as important as selecting the right model"
mathjax: "true"
---
## Introduction to Optimization strategies used in Neural Networks

### Background

As one starts to use Neural Networks in their data models, he will inevitably encounter code of form like this:
<figure style="width: 700px" class="align-center">
  <img src="/images/Machine%20learning/optimization.png" alt="">
</figure> 

One might be quickly puzzled by the 3 terms `optimizer`, `adam` and `sparse_categorical_crossentropy` here. The first 2 are part of this blog's focus, which is about the optimization strategy applied in a Neural Network execution, and the `sparse_categorical_crossentropy` is a loss function used to help with the optimization. 

To understand the relevance of optimizer, one must first understand how an NN is trained. During the training of an NN, the weights of each neuron keeps getting updated so that the `loss` can be minimized. However, randomly updating the weights is not really feasible as there are hundreds of thousands of weights. Hence our smart scientists came up with a backward propagation (BP) algorithm for updating the weights. One may learn more about BP [here](https://towardsdatascience.com/how-does-back-propagation-in-artificial-neural-networks-work-c7cad873ea7). Behind BP we now require the `optimizer` to facilitate the updating of weights in each iteration.

Right below we discuss a few most commonly used optimizers:
### Gradient Descent

Gradient Descent is the most basic optimization strategy which is based on the first order derivative of a loss function. The first order derivative serves as a guide on the direction to modify the weight so as to minimize the loss function. We've discussed its variant in details in an earlier post. To refresh our memory and make this blog more coherent, let's quickly recap here.

Analytic form: $\theta = \theta - \alpha * \nabla J(\theta)$

Characteristics of Gradient Descent include:
* It’s used heavily in linear regression and classification algorithms.
* Easy computation and implementatoin (`Pros`)
* May trap at local minima (`Cons`)
* Weights are changed only after calculating gradient on the whole dataset. So, if the dataset is too large then the convergence may take very long time (`Cons`)
* Requires large memory to calculate gradient on the whole dataset (`Cons`)

### Stochastic Gradient Descent

Gradient Descent has the problem of calculate gradient on the whole dataset in each itearation for weight update. Here Stochastic Gradient Descent aims to resolve this issue by processing data in random batches.

As the model parameters are frequently updated parameters have high variance and fluctuations in loss functions at different intensities.

Analytic form: $\theta = \theta - \alpha * \nabla J(\theta; x_i; y_i)$

Characteristics of SGD include:
* The learning rate needs to be updated in each iteartion to aviod over-fitting
* Faster convergence rate and less memory used (`Pros`)
* High variance in model parameters. (`Cons`)
* May continue to run even when global minima is achieved. (`Cons`)
* To reduce the variance we further have the `mini-batch Gradient Descent` which divides the data into mutiple batches and updates the model parameters after every batch (vs 1 data entry per update in SGD). 


__In general__, Gradient Descent method has the challenge of 
* Choosing an optimum value of the learning rate. If the learning rate is too small than gradient descent may take ages to converge. 
* Have a constant learning rate for all the parameters. There may be some parameters which we may not want to change at the same rate.
* May get trapped at local minima.

### Momentum

Momentum was invented for reducing high variance in SGD and softens the convergence. It takes advantage of information from previous directions via a formula $V(t) = \gamma V(t) + \alpha * \nabla J(\theta)$

Analytic form: $\theta = \theta - V(t)$

Characteristics of Momentum include: 
* The momentum term $\gamma$ is usually set to 0.9 or a similar value.
* Faster Convergence and smaller variance (`pros`)
* Less Oscilliation & more smooth shifting of direction (`pros`)

### Adagrad
Often, the learning rate $\alpha$ of the optimizer is a constant. However, one may expect the optimizer to explore faster at the start and slower at the end to quickly converge to an optimum. Hence the learning rate may subject to change as iteration goes. `Adagrad` aims to achieve such effect. If we use low learning rates for parameters associated with most frequently occurring features, and high learning rates for parameters associated with infrequent features. We can get a good model.

Analytic form:
\begin{align}
  \theta_t = \theta_{t-1} - \frac{\alpha}{\sqrt{\epsilon I + diag(G_t)}}\cdot g_t
\end{align}
where $g_t = [g_{t,i}], \; g_{t,i} =  \nabla_{\theta} J(\theta_{t,i})$ and $G_t = \sum_{n = 1}^{t} g_n$
Here $\epsilon$ is a smoothing term that avoids division by zero (usually on the order of 1e−8)

Characteristics of Adagrad include: 
* Learning rate changes for each training parameter.
* Don’t need to manually tune the learning rate. (`pros`)
* Able to train and performs well on sparse data. (`pros`)
* Computationally expensive as a need to calculate the second order derivative. (`cons`)
* Learning rate is monotone decreasing as iteration $t$ increases. (`cons`)

### AdaDelta
It is an extension of __AdaGrad__ which tends to remove the *decaying learning Rate* problem of it. Instead of accumulating all previously squared gradients, Adadelta limits the window of accumulated past gradients to some fixed size $w$. In this optimizer, exponentially moving average is used rather than the sum of all the gradients.

By the idea above, we reducing the window size of $G_t = \sum_{n = 1}^{t} g_n$ from $t$ to $w$: $G_t^w = \sum_{n = t-w+1}^{w} g_n$.

Analytic form:
\begin{equation}
    \theta_t = \theta_{t-1} - \frac{\alpha}{\sqrt{\epsilon I + diag(G_t^w)}}\cdot g_t
\end{equation}

Characteristics of AdaDelta include: 
* Learning rate does not decay necessarily (`pros`)
* More computationally expensive as expectation is involved (`cons`)

### RMSProp

The RMSProp algorithm full form is called Root Mean Square Prop, which is an adaptive learning rate optimization algorithm proposed by Geoff Hinton.

RMSProp is another strategy that tries to resolve `Adagrad`’s radically diminishing learning rates problem by using a moving average of the squared gradient. It utilizes the magnitude of the recent gradient descents to normalize the gradient.

While Adagrad accumulates all previous gradient squares, RMSprop just calculates the corresponding average value, so it can eliminate the problem of quickly dropping of learning rate of the Adagrad.

By the idea above, we replace the $G_t = \sum_{n = 1}^{t} g_n$ with an expectation formula: 
  
  $$\mathbb{E}[g_t^2] = \gamma \mathbb{E}[g_{t-1}^2] + (1-\gamma)g_t^2 .$$

Analytic form:
\begin{equation}
    \theta_t = \theta_{t-1} - \frac{\alpha}{\sqrt{\epsilon I + \mathbb{E}[g_t^2])}}\cdot g_t
\end{equation}

__Conclusion__ for the dynamic learning rate optimizer:
* Good for sparse data
* Be careful of the diminishing speed of learning rate
* More expensive computationally in general

### Adam

Adam (Adaptive Moment Estimation) works with momentums of first and second order. The intuition behind the Adam is that we don’t want to roll so fast just because we can jump over the minimum, we want to decrease the velocity a little bit for a careful search. In addition to storing an exponentially decaying average of past squared gradients like AdaGrad, Adam also keeps an exponentially decaying average of past gradients M(t). In summary, Adam can be looked at as a combination of RMSprop and Stochastic Gradient Descent with momentum.

Note that although the name `momentum` looks fancy, the terms we need to consider are just first and second order momentum, which are essentially `mean` and `variance` of the gradients. Afterwards, we can consider 2 terms $m_t$ and $v_t$ as follows

Hence the formula is as follows:

$$ m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \; \; v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

These 2 terms are used to approximate the first and second moments, that is:

$$ \mathbb{E}[m_t] \approx  \mathbb{E}[g_t]\;\;;\; \mathbb{E}[v_t] \approx  \mathbb{E}[g_t^2]$$

Although we have $\mathbb{E}[\cdot]$ above, theorem suggests that we can use the observed $m_t$ and $v_t$ to approximate $\mathbb{E}[g_t]$ and $\mathbb{E}[g_t^2]$ directly.

After bias correction, we derive the terms 

$$\hat{m_t} = \frac{m_t}{1-\beta_1^2}\;\; ;\; \hat{v_t} = \frac{v_t}{1-\beta_2^2}$$

Here $\beta_1$ and $\beta_2$ have really good default values of 0.9 and 0.999 respectively.

Finally the update formula is just $\theta_t = \theta_{t-1}  - \alpha * \frac{\hat{m_t}}{\sqrt{\hat{v_t}} + \epsilon} $


## Code implementation
**To Be Updated**
