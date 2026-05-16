---
title: "Deep Learning Training: A Practical Guide"
excerpt: "A practical guide to optimizer choice, learning-rate schedules, stability, memory pressure, throughput, checkpointing, and experiment management during deep learning training."
date: 2024/02/19
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

Training is where research ideas meet hardware limits. A model can be architecturally clever and still fail because the learning rate is unstable, the data loader is slow, the checkpoint cannot resume, or the GPU memory budget is wrong.

This post focuses on the practical decisions inside a deep learning training run:

- How to choose optimizers and schedules.
- How to detect instability early.
- How to reduce memory use without hiding accuracy regressions.
- How to improve throughput before adding more machines.
- How to make training resumable and observable.

I assume the basic pipeline already exists: dataset, model, loss function, evaluation metric, and a working single-device training loop.

## Start With the Contract

Before tuning anything, write down the training contract:

- **Task:** classification, ranking, generation, retrieval, regression, or another objective.
- **Metric:** the validation metric that decides whether a run improved.
- **Slices:** important subgroups or scenarios where regressions are unacceptable.
- **Budget:** maximum GPU hours, wall-clock time, and cloud cost.
- **Target:** research checkpoint, production candidate, fine-tuned adapter, or benchmark result.
- **Resume rule:** what state must be saved to continue after interruption.

This contract prevents a common failure mode: optimizing training speed while forgetting what the trained model must actually prove.

## Optimizer Choice

Most training runs start with one of these optimizers:

- **SGD with momentum:** simple, memory efficient, and still strong for many vision tasks.
- **Adam:** adaptive and forgiving, useful for many deep learning settings.
- **AdamW:** Adam with decoupled weight decay, a common default for transformers.
- **Adafactor:** memory-efficient for very large models, but more sensitive to configuration.

AdamW is often the pragmatic default for modern transformer training:

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.95),
    weight_decay=0.1,
)
```

Optimizer choice affects memory. Adam-style optimizers keep extra state, commonly first and second moment estimates, so optimizer state can be much larger than the model weights alone. If memory is tight, optimizer choice is part of the memory plan, not only the convergence plan.

## Learning-Rate Schedules

The learning rate is usually more important than the optimizer name. A good schedule controls both early instability and late convergence.

Common schedules:

- **Warmup plus cosine decay:** a strong default for transformer-style training.
- **Linear warmup plus linear decay:** common in fine-tuning.
- **Step decay:** simple and useful when training behavior is well understood.
- **Reduce on plateau:** useful when validation metrics are noisy but meaningful.
- **One-cycle:** can work well when you want fast convergence and have a tuned range.

Warmup matters because early gradients can be volatile. For large models, skipping warmup can produce immediate loss spikes or NaNs.

```python
from transformers import get_cosine_schedule_with_warmup


scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=1_000,
    num_training_steps=50_000,
)
```

Track the learning rate in your experiment logs. It is hard to debug training curves without knowing which schedule was active.

## Stability Signals

Training instability is easier to fix when caught early. Log these signals:

- Training loss and validation loss.
- Validation metric and important slice metrics.
- Learning rate.
- Gradient norm.
- Weight norm or selected layer norms.
- Number of NaN or Inf events.
- GPU memory and step time.
- Data loading time.

Gradient clipping is a common guardrail:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Do not use clipping to hide a broken run. If clipping activates constantly, the learning rate, loss scaling, data, or model initialization may be wrong.

## Effective Batch Size

The effective batch size is:

```text
per_device_batch_size * number_of_devices * gradient_accumulation_steps
```

Changing this value can change convergence. If you use more GPUs and keep the same per-device batch size, the effective batch size grows. You may need to retune learning rate, warmup, regularization, or number of training steps.

Gradient accumulation is useful when the target batch size does not fit in memory:

```python
accumulation_steps = 4

for step, batch in enumerate(train_loader):
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        loss = model(**batch).loss
        loss = loss / accumulation_steps

    loss.backward()

    if (step + 1) % accumulation_steps == 0:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
```

The tradeoff is wall-clock time. Gradient accumulation reduces memory pressure, but it does not create the same parallelism as a larger physical batch.

## Memory Reduction

When a run hits out-of-memory errors, use the least invasive fix first.

### Smaller Micro-Batches

Reducing per-device batch size is the simplest fix. Combine it with gradient accumulation if you need to preserve effective batch size.

### Mixed Precision

Mixed precision reduces activation memory and can improve throughput. BF16 is often the preferred choice when hardware supports it because it has a wider exponent range than FP16.

```python
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    loss = model(**batch).loss
```

FP16 may require gradient scaling:

```python
scaler = torch.amp.GradScaler("cuda")

with torch.amp.autocast("cuda", dtype=torch.float16):
    loss = model(**batch).loss

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Activation Checkpointing

Activation checkpointing saves memory by not storing every intermediate activation during the forward pass. During backward, selected parts of the forward computation are recomputed.

This trades compute for memory. It is often a good fit when memory is the blocker and extra compute is acceptable.

```python
from torch.utils.checkpoint import checkpoint


def forward(self, x):
    x = checkpoint(self.block1, x, use_reentrant=False)
    x = checkpoint(self.block2, x, use_reentrant=False)
    return self.head(x)
```

### Sharding and Offloading

When optimizer state, gradients, or parameters are too large, use sharding strategies such as FSDP or ZeRO-style training. CPU offload can help fit the run, but it increases communication cost and can slow training substantially.

Use sharding after the single-device loop is correct. Distributed memory tricks make debugging harder.

### Quantization

Quantization is usually a deployment or fine-tuning technique, but low-bit loading can also make large-model adapter training possible. Evaluate carefully, because low-bit training changes the numerical behavior of the model.

For a deeper overview, see [Quantization in Deep Learning](/writing/blogs/mlops/quantization/).

## Throughput Improvements

Before adding GPUs, make sure one GPU is not waiting on the rest of the system.

Check:

- Data loader worker count and prefetching.
- Tokenization or image decoding inside the training step.
- Host-to-device transfer time.
- GPU utilization.
- Sequence length distribution.
- Padding waste.
- Checkpoint save time.
- Evaluation frequency.

PyTorch Profiler is useful when intuition is not enough:

```python
import torch.profiler


with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
) as prof:
    train_one_epoch()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

`torch.compile` can improve throughput for some models, especially when the workload is compute-bound and graph breaks are limited:

```python
model = torch.compile(model)
```

Do not assume it will always help. Measure step time before and after.

## Distributed Training

Distributed training is the right answer when the bottleneck is either throughput or memory and simpler fixes are exhausted.

Use this progression:

1. Single GPU baseline.
2. Mixed precision and data pipeline cleanup.
3. Single-node DDP for throughput.
4. Activation checkpointing for memory.
5. FSDP or ZeRO-style sharding for larger models.
6. Multi-node only when one node is not enough.

For details, see [Understanding Distributed Training in Deep Learning](/writing/blogs/mlops/distributed-training/).

## Checkpointing

A checkpoint should contain enough state to resume training, not just enough state to run inference.

Save:

- Model state.
- Optimizer state.
- Scheduler state.
- Mixed precision scaler state, if used.
- Epoch and step.
- Random number generator state when reproducibility matters.
- Config and data version.

```python
torch.save(
    {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": global_step,
        "config": config,
    },
    "checkpoint.pt",
)
```

For distributed training, use distributed checkpoint tooling or save only on rank 0 when appropriate. Test resume before running an expensive job.

## Managing the Run

A training run should leave behind a record that another engineer can understand.

Log:

- Config.
- Git commit.
- Dataset version.
- Environment or container image.
- Hardware.
- Metrics and slice metrics.
- Evaluation artifacts.
- Checkpoints.
- Failure logs.

This is where post-training workflow begins: experiment tracking, model registry, serving, and monitoring. See [MLOps Post-Training Considerations](/writing/blogs/mlops/ml-post-training/).

## Closing

Good training is disciplined iteration. Start with the simplest reliable run, add complexity only when the bottleneck is visible, and measure each change against the training contract.

The best training system is not the most elaborate one. It is the one that helps you make valid decisions quickly, recover from failures, and know why a model improved.

## References

- [PyTorch Automatic Mixed Precision](https://docs.pytorch.org/docs/2.12/amp.html)
- [PyTorch Activation Checkpointing](https://docs.pytorch.org/docs/2.12/checkpoint.html)
- [PyTorch Profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [PyTorch torch.compile](https://docs.pytorch.org/docs/2.12/generated/torch.compile.html)
- [Lightning Trainer](https://lightning.ai/docs/pytorch/stable/common/trainer.html)
