---
title: "Neural Network Applied: Optimizer Selection"
excerpt: "A practical guide to choosing SGD, momentum, RMSProp, Adam, AdamW, and related optimizers for neural network training."
date: 2023/12/15
categories:
  - Blogs
tags:
  - Deep Learning
  - Optimization
mathjax: true
toc: true
---

## Background

An optimizer is the part of training that updates model parameters after gradients are computed. The loss function says what the model should improve. Backpropagation computes gradients. The optimizer decides how to use those gradients.

The simplest update is gradient descent:

$$
\theta_{t+1} = \theta_t - \alpha \nabla J(\theta_t)
$$

Here $\theta$ is the parameter vector, $\alpha$ is the learning rate, and $J$ is the objective function.

In neural networks, the practical question is not "Which optimizer is theoretically best?" It is "Which optimizer is stable, efficient, and well matched to this model and data?"

## Gradient Descent and Mini-Batch SGD

Full-batch gradient descent computes gradients over the whole dataset before each update. That is usually too expensive for deep learning.

Mini-batch stochastic gradient descent (SGD) updates the model after each mini-batch:

$$
\theta_{t+1} = \theta_t - \alpha \nabla J(\theta_t; x_b, y_b)
$$

SGD is simple and memory efficient. It often generalizes well, especially in vision workloads, but it can require careful learning-rate tuning and scheduling.

Use SGD when:

- You want a simple, memory-light optimizer.
- The model and task are known to work well with SGD.
- You can afford learning-rate tuning.
- Generalization matters more than fast early progress.

## Momentum

Momentum smooths updates by accumulating a velocity term:

$$
v_t = \gamma v_{t-1} + \alpha \nabla J(\theta_t)
$$

$$
\theta_{t+1} = \theta_t - v_t
$$

Momentum helps reduce oscillation and can accelerate progress in consistent descent directions. A common value is $\gamma = 0.9$.

Use SGD with momentum as a strong baseline when plain SGD is too noisy.

## AdaGrad

AdaGrad adapts the learning rate for each parameter based on the sum of squared historical gradients:

$$
\theta_{t+1} =
\theta_t -
\frac{\alpha}{\sqrt{G_t + \epsilon}} g_t
$$

It works well for sparse features because frequently updated parameters receive smaller effective learning rates. Its weakness is that the accumulated denominator only grows, so learning can slow too much over long training.

Use AdaGrad when sparse features are central and the training run is not extremely long.

## RMSProp

RMSProp fixes AdaGrad's aggressively shrinking learning rate by using an exponential moving average of squared gradients:

$$
E[g^2]_t = \rho E[g^2]_{t-1} + (1 - \rho)g_t^2
$$

$$
\theta_{t+1} =
\theta_t -
\frac{\alpha}{\sqrt{E[g^2]_t + \epsilon}} g_t
$$

RMSProp adapts learning rates without letting the denominator grow forever. It is historically important and still useful, though Adam-style optimizers are more common defaults today.

## Adam

Adam combines momentum-like first moment estimates with RMSProp-like second moment estimates:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2
$$

Bias-corrected estimates are then used for the update:

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}
$$

$$
\hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$

$$
\theta_{t+1} =
\theta_t -
\alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
$$

Adam is a good default when you need fast, stable progress and do not want to hand-tune as much as SGD. Common defaults are $\beta_1 = 0.9$, $\beta_2 = 0.999$, and $\epsilon = 10^{-8}$.

## AdamW

AdamW decouples weight decay from the Adam gradient update. This matters because L2 regularization and weight decay are not equivalent under adaptive optimizers.

AdamW is a common default for transformer training and fine-tuning:

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.95),
    weight_decay=0.1,
)
```

Use AdamW when:

- Training transformers.
- Fine-tuning pretrained neural networks.
- Weight decay is part of the regularization plan.
- You want a strong modern default.

## Adafactor

Adafactor reduces optimizer memory by factorizing second-moment estimates. This can be useful for very large language models where AdamW optimizer state is expensive.

The tradeoff is that Adafactor can be more sensitive to configuration. Use it when optimizer memory is a real bottleneck, not just because it exists.

## Optimizer Selection Heuristics

Use this as a starting point:

- **Classical deep vision baseline:** SGD with momentum or AdamW.
- **Transformer training or fine-tuning:** AdamW.
- **Very large model with optimizer-memory pressure:** Adafactor or sharded AdamW.
- **Sparse features:** AdaGrad or an adaptive optimizer.
- **Small noisy dataset:** AdamW with careful weight decay, early stopping, and validation monitoring.
- **Memory-constrained training:** consider SGD, Adafactor, optimizer sharding, or offload.

Then tune the learning rate. Optimizer choice matters, but a bad learning rate can make any optimizer look broken.

## Scheduler Pairing

Optimizers and schedules should be chosen together.

Common pairings:

- SGD with momentum plus step decay or cosine decay.
- AdamW plus warmup and cosine decay.
- AdamW plus linear warmup and linear decay for fine-tuning.
- Reduce-on-plateau when validation metrics are meaningful and training is slower.

Always log the learning rate. It is part of the experiment state.

## Practical Debugging

If training is unstable:

- Lower the learning rate.
- Add warmup.
- Check data and labels.
- Check for NaNs or Infs.
- Track gradient norm.
- Clip gradients if spikes are rare.
- Confirm loss reduction on a tiny batch.
- Try BF16 or FP32 if FP16 is unstable.

If training is too slow:

- Profile the data loader.
- Increase batch size if memory allows.
- Use mixed precision.
- Try `torch.compile` if the model is a fit.
- Improve the scheduler before changing architectures.

## Closing

Optimizer selection is a practical decision. Start with a strong default, tune the learning rate and schedule, measure stability, and only switch optimizers when the symptoms point to a real bottleneck.

For a broader training workflow, see [Deep Learning Training: A Practical Guide](/writing/blogs/mlops/ml-training/).

## References

- [PyTorch Optimizers](https://docs.pytorch.org/docs/2.12/optim.html)
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
