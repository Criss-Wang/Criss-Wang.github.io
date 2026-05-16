---
title: "Understanding Distributed Training in Deep Learning"
excerpt: "A practical map of data parallelism, model sharding, pipeline parallelism, launch tools, and the bottlenecks that usually decide whether distributed training is worth it."
layout: post
date: 2024/03/04
categories:
  - Blogs
tags:
  - Distributed Systems
  - Deep Learning
mathjax: true
toc: true
---

## Introduction

Distributed training is not just "training on more GPUs." It is a set of tradeoffs between memory, compute, communication, engineering complexity, and failure recovery.

The motivation is simple:

- A model may not fit on one device.
- A dataset may take too long to train on one device.
- A training run may need more throughput to make iteration possible.
- A production team may need fault tolerance, repeatability, and predictable cost.

The hard part is that every form of parallelism moves the bottleneck somewhere else. Data parallelism increases gradient communication. Tensor and pipeline parallelism reduce memory pressure but add scheduling and synchronization overhead. Offloading saves GPU memory but increases CPU-GPU or network traffic.

This post is a practical map of the major strategies and the mistakes that usually matter in real projects.

## What Gets Parallelized

Most distributed training systems combine several forms of parallelism.

### Data Parallelism

Data parallelism replicates the model on each worker and splits batches across workers. Each process computes gradients on its local mini-batch, then the workers synchronize gradients, usually with an all-reduce.

This is the first strategy to try when the model fits on a single GPU and the bottleneck is throughput.

In PyTorch, the standard tool is `DistributedDataParallel` (DDP). One important detail: DDP synchronizes gradients, but it does not split the input data for you. You still need a `DistributedSampler` or equivalent data sharding.

```python
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler


def setup_distributed() -> int:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return local_rank


def train(model, dataset, optimizer, loss_fn, epochs: int) -> None:
    local_rank = setup_distributed()

    sampler = DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=32, sampler=sampler, num_workers=4)

    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank])

    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        for inputs, labels in loader:
            inputs = inputs.to(local_rank, non_blocking=True)
            labels = labels.to(local_rank, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(inputs), labels)
            loss.backward()
            optimizer.step()

    dist.destroy_process_group()
```

Launch a single-node run with `torchrun`:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py
```

### Fully Sharded Data Parallelism

Fully Sharded Data Parallelism (FSDP) shards model parameters, gradients, and optimizer state across workers. It is useful when the model is too large for regular DDP, especially with optimizers like AdamW where optimizer state can be several times larger than the raw parameters.

FSDP works by gathering parameter shards when a module needs them, computing forward or backward work, and then freeing or re-sharding the full parameters. This saves memory at the cost of more communication.

```python
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


dist.init_process_group(backend="nccl")
model = MyModel().to(torch.cuda.current_device())
model = FSDP(model)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
```

Use FSDP when memory is the blocker. If the model already fits comfortably on one GPU, plain DDP is usually simpler and often faster.

### Tensor Parallelism

Tensor parallelism splits individual tensor operations across devices. Instead of giving each GPU a full copy of a layer, the layer's matrix multiplications are partitioned across GPUs.

This matters for very large transformer models where a single attention or feed-forward block may be too large or too slow on one device. It often requires model-aware implementation, so it is less plug-and-play than DDP.

### Pipeline Parallelism

Pipeline parallelism splits the model into stages. Stage 1 runs the first set of layers, stage 2 runs the next set, and so on. Micro-batches keep the pipeline busy.

The main risk is idle time. If one stage is slower than the others, the faster stages wait. That idle period is called a pipeline bubble. Good pipeline training depends on balanced stage assignment, sensible micro-batch size, and careful scheduling.

### Offloading

Offloading moves parameters, gradients, or optimizer state from GPU memory to CPU memory or storage when they are not immediately needed. It can make otherwise impossible runs feasible, but it almost always increases latency.

Use offloading when memory is the hard limit. Do not expect it to be free.

## Tooling Choices

### Native PyTorch

Native PyTorch gives the most control. Use it when you need to understand the system, customize training deeply, or debug low-level distributed behavior.

The main building blocks are:

- `torchrun` for launching distributed processes.
- `torch.distributed` for process groups and collectives.
- `DistributedDataParallel` for replicated data parallel training.
- `FullyShardedDataParallel` for sharded training.
- `torch.distributed.checkpoint` for distributed checkpointing.

### Lightning

Lightning is useful when you want a higher-level training loop with standard strategies:

```python
import lightning as L


trainer = L.Trainer(
    accelerator="gpu",
    devices=4,
    strategy="ddp",
    precision="bf16-mixed",
    max_epochs=3,
)
trainer.fit(model, datamodule=data)
```

Switching `strategy="ddp"` to `strategy="fsdp"` can be a good starting point for memory-constrained runs, but the model still needs careful validation.

### Hugging Face Accelerate

Accelerate is helpful when you want to keep a mostly plain PyTorch loop while letting the library handle device placement and distributed wrapping.

```python
from accelerate import Accelerator


accelerator = Accelerator()
model, optimizer, train_loader, scheduler = accelerator.prepare(
    model,
    optimizer,
    train_loader,
    scheduler,
)

for batch in train_loader:
    optimizer.zero_grad()
    loss = loss_fn(model(batch["inputs"]), batch["labels"])
    accelerator.backward(loss)
    optimizer.step()
    scheduler.step()
```

Launch it with:

```bash
accelerate config
accelerate launch train.py
```

## Common Bottlenecks

### Communication Overhead

More workers do not automatically mean faster training. At some point, gradient synchronization, parameter gathering, or pipeline communication can dominate the step time.

Watch these signals:

- GPU utilization drops while network utilization rises.
- Step time stops improving as you add GPUs.
- Small batches scale poorly.
- Profiler traces show long all-reduce or all-gather blocks.

Common fixes include larger per-device batches, gradient accumulation, faster interconnects, mixed precision, communication overlap, and reducing the frequency or size of synchronization.

### Data Loading

Distributed training can expose a data pipeline that was "fast enough" on one GPU. If workers wait for batches, scaling compute will not help.

Check:

- Data loader worker count.
- Object storage latency.
- On-the-fly decoding or tokenization.
- Dataset sharding.
- Host-to-device transfer time.
- Whether `pin_memory` and `non_blocking` transfers help.

### Memory

GPU memory is consumed by parameters, gradients, activations, optimizer state, temporary buffers, and framework overhead. For AdamW, optimizer state can be a major part of memory use.

The usual memory levers are:

- Mixed precision, often BF16 when hardware supports it.
- Activation checkpointing.
- Smaller micro-batches plus gradient accumulation.
- FSDP or ZeRO-style sharding.
- CPU offload.
- Quantization or smaller model variants.

### Fault Tolerance

Long distributed jobs fail. Machines preempt, network links flap, and workers hit data-specific errors.

Make failure recovery boring:

- Save checkpoints regularly.
- Save optimizer, scheduler, scaler, and RNG state when reproducibility matters.
- Test resume before the expensive run.
- Log rank-specific errors.
- Keep the same container image across workers.
- Store config, git commit, data version, and launch command.

## Mistakes To Avoid

### Changing the Model After Wrapping

Wrap the model with DDP or FSDP after the model structure is final. Changing parameters afterward can break gradient synchronization assumptions.

### Forgetting `sampler.set_epoch`

With `DistributedSampler`, call `sampler.set_epoch(epoch)` each epoch so shuffling changes across epochs consistently.

### Comparing Runs With Different Effective Batch Sizes

The effective batch size is:

```text
per_device_batch_size * number_of_processes * gradient_accumulation_steps
```

If this changes, learning rate, warmup, convergence, and generalization can change too.

### Ignoring Pipeline Balance

Pipeline parallelism needs roughly balanced stages. If one stage is much slower, the rest of the pipeline waits. Profile the model before deciding where to split it.

### Treating Multi-Node as "Single-Node, But Bigger"

Multi-node training adds rendezvous, network topology, firewalls, hostnames, clock drift, shared storage, rank assignment, and failure modes. Test a tiny job across the same topology before paying for a full run.

## A Practical Decision Path

Start with the least complex strategy that addresses the actual bottleneck:

1. One GPU baseline: confirm the model, data, loss, and metrics work.
2. Single-node DDP: use this when the model fits but training is too slow.
3. Mixed precision and activation checkpointing: use these before adding complex sharding.
4. FSDP or ZeRO-style sharding: use when model, gradients, or optimizer state do not fit.
5. Tensor or pipeline parallelism: use when individual layers or model stages need to be split.
6. Multi-node training: use when one machine is not enough and you can afford the operational complexity.

For most teams, the best first distributed setup is boring: DDP, `torchrun`, a pinned container image, strong logging, and tested checkpoint resume.

## References

- [PyTorch DistributedDataParallel](https://docs.pytorch.org/docs/2.12/generated/torch.nn.parallel.DistributedDataParallel.html)
- [PyTorch torchrun](https://docs.pytorch.org/docs/2.12/elastic/run.html)
- [PyTorch FullyShardedDataParallel](https://docs.pytorch.org/docs/2.9/fsdp.html)
- [Lightning strategies](https://lightning.ai/docs/pytorch/stable/extensions/strategy.html)
- [Hugging Face Accelerate migration guide](https://huggingface.co/docs/accelerate/main/en/basic_tutorials/migration)
