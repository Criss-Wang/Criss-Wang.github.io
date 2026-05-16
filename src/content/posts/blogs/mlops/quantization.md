---
title: "Quantization in Deep Learning"
excerpt: "A practical guide to post-training quantization, quantization-aware training, mixed precision, and low-bit model deployment."
date: 2024/02/23
categories:
  - Blogs
tags:
  - Deep Learning
  - Distributed Training
layout: post
mathjax: true
toc: true
---

## Introduction

Quantization reduces the numerical precision used to store or compute model values. Instead of keeping every weight and activation in `float32`, a deployed model might use `int8`, `float16`, `bfloat16`, `int4`, or a specialized low-bit format.

The goal is usually one of four things:

- Reduce model size.
- Reduce memory bandwidth.
- Improve inference latency.
- Make a model fit on cheaper or smaller hardware.

The tradeoff is accuracy. Quantization changes the values the model uses, so every quantized model needs task-level evaluation, not just a smaller file size.

## The Core Idea

Uniform affine quantization maps a floating-point value to an integer range:

```text
q = clamp(round(x / scale + zero_point), q_min, q_max)
x_hat = scale * (q - zero_point)
```

The model runs with the quantized value `q`, then dequantizes to an approximation `x_hat` when needed. The choice of `scale` and `zero_point` determines how much information is preserved.

Two design choices matter a lot:

- **Per-tensor vs per-channel:** per-channel quantization gives each output channel its own scale and usually preserves accuracy better for weights.
- **Symmetric vs asymmetric:** symmetric quantization is simpler, while asymmetric quantization can represent shifted distributions more efficiently.

## Common Quantization Strategies

### Dynamic Quantization

Dynamic quantization is often the easiest first step. Weights are quantized ahead of time, while activation ranges are determined dynamically at inference.

It is commonly useful for models with large `Linear` layers, especially CPU inference. It is less useful when the bottleneck is not matrix multiplication or when the deployment backend does not accelerate the chosen quantized operators.

```python
import torch
from torch.ao.quantization import quantize_dynamic


model_fp32 = MyModel().eval()
model_int8 = quantize_dynamic(
    model_fp32,
    {torch.nn.Linear},
    dtype=torch.qint8,
)
```

Dynamic quantization is attractive because it needs little calibration data and no retraining. The downside is that it may leave activation memory and some compute in higher precision.

### Post-Training Static Quantization

Static quantization quantizes both weights and activations. Because activation ranges depend on real inputs, the model needs a calibration pass on representative data.

The rough workflow is:

1. Train the original model.
2. Put it in evaluation mode.
3. Attach a quantization configuration.
4. Run calibration data through the prepared model.
5. Convert the prepared model to a quantized model.
6. Evaluate it on real validation slices.

```python
import torch
from torch.ao.quantization import convert, get_default_qconfig, prepare


model = MyModel().eval()
model.qconfig = get_default_qconfig("x86")

prepared = prepare(model)
with torch.no_grad():
    for inputs, _ in calibration_loader:
        prepared(inputs)

quantized_model = convert(prepared)
```

Static quantization can reduce latency more than dynamic quantization, but it is more sensitive to calibration quality. If the calibration set misses important input ranges, production accuracy can suffer.

### Quantization-Aware Training

Quantization-aware training (QAT) simulates quantization during training. The model learns while seeing fake-quantized weights and activations, then gets converted to a quantized model after fine-tuning.

Use QAT when post-training quantization produces unacceptable accuracy loss and retraining is feasible.

```python
from torch.ao.quantization import convert, get_default_qat_qconfig, prepare_qat


model = MyModel().train()
model.qconfig = get_default_qat_qconfig("x86")
qat_model = prepare_qat(model)

for inputs, labels in train_loader:
    optimizer.zero_grad(set_to_none=True)
    loss = loss_fn(qat_model(inputs), labels)
    loss.backward()
    optimizer.step()

qat_model.eval()
quantized_model = convert(qat_model)
```

QAT is more expensive than post-training quantization, but it gives the model a chance to adapt to quantization noise.

### Weight-Only and Low-Bit Quantization

Large language models often use weight-only quantization, such as 8-bit or 4-bit weights, because model weights dominate memory. Activations may remain in BF16 or FP16 while weights are stored in a lower-bit format.

QLoRA popularized a practical pattern: load a base model in 4-bit precision, freeze most of it, and train small adapter layers. Formats such as NF4 are designed for normally distributed neural network weights. Double quantization can further reduce memory by quantizing quantization constants.

The key point: low-bit LLM quantization is usually backend-specific. Always check the exact library, hardware, and serving stack before assuming speedups.

## Mixed Precision Is Related, But Different

Mixed precision training uses lower precision for parts of training while preserving enough high-precision state to keep optimization stable. It is not the same as deployment quantization.

Typical training choices:

- **FP16:** fast on many GPUs but may need gradient scaling.
- **BF16:** wider exponent range, often more stable, and commonly preferred when hardware supports it.
- **FP32 master weights or optimizer state:** still used in many training setups for stability.

In PyTorch, automatic mixed precision is usually handled with `torch.amp`:

```python
import torch


scaler = torch.amp.GradScaler("cuda")

for inputs, labels in train_loader:
    optimizer.zero_grad(set_to_none=True)

    with torch.amp.autocast("cuda", dtype=torch.float16):
        loss = loss_fn(model(inputs), labels)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

For BF16 training, gradient scaling is often unnecessary:

```python
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    loss = loss_fn(model(inputs), labels)
```

## Choosing a Quantization Path

Use this order unless you have a strong reason not to:

1. Establish a full-precision baseline with task-level metrics.
2. Try dynamic quantization if the model is CPU-bound or mostly linear layers.
3. Try static post-training quantization if deployment supports int8 activation kernels.
4. Use QAT if static quantization loses too much accuracy.
5. Use 4-bit or weight-only methods for large model memory pressure, especially adapter fine-tuning or inference.
6. Benchmark on the target hardware, not just a development laptop.

The evaluation should include:

- Overall metric change.
- Slice-level metric change.
- Latency distribution, not only average latency.
- Memory footprint.
- Throughput under realistic batch size.
- Failure cases where quantization changes the predicted class or generated output.

## Practical Warnings

Quantization can fail quietly. A model can look smaller while becoming slower if the backend falls back to dequantized operators. It can also pass aggregate validation while failing on rare but important slices.

Watch for:

- Unsupported operators.
- Backend-specific behavior between CPU, CUDA, mobile, and edge runtimes.
- Calibration data that is too small or too clean.
- Outlier activation channels.
- Accuracy regressions hidden by only tracking a headline metric.
- Drift after deployment, because input ranges can move after calibration.

Quantization is best treated as a deployment optimization with its own validation plan, not as a final compression button.

## References

- [PyTorch Quantization API Reference](https://docs.pytorch.org/docs/2.12/quantization-support.html)
- [PyTorch Quantization-Aware Training for Large Language Models](https://docs.pytorch.org/blog/quantization-aware-training/)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [NVIDIA Mixed Precision Training](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html)
