## FSDP

1. What is in the GPU memory (x params, FP16)
   1. Params: 2x (fp16 with 2 bytes)
   2. Gradients: 2x
   3. Optimizer (AdamW)
      - Param copy: 4x (float32)
      - Momentum: 4x
      - Variance: 4x
2. How DDP works
   - NCCL: multi-GPU, multi-node **communication** primitives. all-gather, all-reduce, broadcast, reduce-scatter, reduce routines, point-to-point send/receive. High bandwidth, low latency on PCIe and NVLink interconnects
   - All GPUs share same initial weights. Aggregate all gradients in different GPUs and update the weight collectively.
   - Update optimizer state and weights after AllReduce
3. Other methods:
   1. Model Parallelism (split horizontally -> Inefficient: GPU 2 idle if layer 2 is not run)
      ```python
      class model_parallel(nn.Module):
      	def __init__(self):
      		super().__init__()
      		self.layer_1 = nn.Sequential(...)
      		self.layer_2 = nn.Sequential(...)
      		self.layer_1.cuda(0)
      		self.layer_2.cude(1)
      	def forward(self, x):
      		x = x.cuda(0)
      		x = self.layer_1(x)
      		x = x.cuda(1)
      		x = self.layer_2(x)
      		x = ...
      		return x
      ```
   2. Tensor parallelism (split vertically)
   3. Pipeline Parallelism (Mixed data and model parallelism, involves scheduling of data flow)
4. How FSDP works
   1. FSDP unit (vertical splitting), can be:
      - A layer splitted
      - A stage splitted
      - A group of layers splitted
   2. Sharding
      - Storing the FSDP unit on `FlatParameter`
      - Split `FlatParameter` on multiple nodes (after zero padding for divisible property)
   3. All-Gather
      - performed by NCCL
      - gather all parts and sync across all nodes
      - Done before both forward and backwards
      - discard peer parts after forward/backward
   4. Reduce-scatter
      - performed via NCCL
      - Each node gets part of the result of gradient (backward only)
      - Note that All-Reduce is not used coz it broadcast same results to all nodes
      - E.g. Each node `i` has all gradients `G_i1, G_i2, ..., G_in`, after reduce-scatter, each node will have gradient redistributed, with node `i` getting `sum of G_ki`, where k spans from 1 to n
5. Reason to use/not to use FSDP
   1. When to use
      - Model size is too large (not data size)
      - More communication between GPUs
      - Hence trade memory for speed: more GPU memory cost due to communication, however, communication overhead reduced via NCCL acceleration
      - If want to trade speed for memory, see **activation checkpointing**
   1. When not to use
      - For models < 100 million params, consider activation-checkpointing and reversible layers
      - Recommend to use BFloat16 instead of Float16 (Float16 requires ShardedGradScaler)
      - Mixed Precision Training Concern (Package compatibility)
6. PyTorch Implementation

```python
from torch.distributed.fsdp import (
   FullyShardedDataParallel,
   CPUOffload,
)
from torch.distributed.fsdp.wrap import (
   default_auto_wrap_policy,
   enable_wrap,
   wrap
)
import torch.nn as nn

class model(nn.Module):
   def __init__(self):
       super().__init__()
       self.layer1 = nn.Linear(8, 4)
       self.layer2 = nn.Linear(4, 16)
       self.layer3 = nn.Linear(16, 4)

model = DistributedDataParallel(model())
fsdp_model = FullyShardedDataParallel(
   model(),
   fsdp_auto_wrap_policy=default_auto_wrap_policy,
   cpu_offload=CPUOffload(offload_params=True),
)

# Custom wrap
wrapper_kwargs = Dict(cpu_offload=CPUOffload(offload_params=True))
with enable_wrap(wrapper_cls=FullyShardedDataParallel, **wrapper_kwargs):
   fsdp_model = wrap(model())
```

## Distributed Training Common Errors

#### Not pipelining

- Pipeline Parallelism is always something to include. Notice the use of ZeRO-3 also uses pipeline parallelism

#### Not balancing pipeline stages

- There will be some brief periods where either a machine is idle and waiting on the next minibatch from the previous machine or takes longer than other machines to execute its computation, thus slowing down the pipeline.
- You should ideally construct your pipeline such that each machine does as close to the same amount of computation as possible. This means timing how long it takes data to get through different layers in the model, timing how long forward and backward propagation takes for each model partition, and ensuring roughly equivalent data sizes across mini-batches. This is critical for optimizing pipeline efficiency.
- To achieve this, setting up profiler like PyTorch Profiler is critical for evaluation of computations done during model training

#### Weight staleness

- When model training is pipelined across multiple machines, there is a delay that happens between when the forward computation on data occurs and when the gradients based on that computation are backpropagated to update the model weights. As a result, forward propagation are calculated using weights that aren\'t updated with the latest gradients.
- Solution: **weight stashing**
  A system “maintains multiple versions of a model’s weights, one for each minibatch.” After the completion of each forward pass, the system can store a model’s weights as part of the state associated with that minibatch. When the time comes for backpropagation, the weights associated with that minibatch are retrieved from the stash and used for the gradient computation. This ensures that the same version of weights are used for the forward and backward pass over a single minibatch of data within a pipelined stage, and statistical convergence is improved.

#### Driver and library inconsistencies between machines

- Containerization / Virtualization using tools like Docker solves the problem

#### Wrong type of Optimizer Update

- Example: Synchronous vs Asynchronous SGD
- Asynchronous SGD ([HogWild](https://people.eecs.berkeley.edu/~brecht/papers/hogwildTR.pdf) as a popular choice) which showed that SGD could be run in parallel, without locks, and without too much effect on algorithm convergence. Asynchronous SGD allows weight updates to proceed without each machine waiting for the other to send their gradients.

#### **Network issues, firewalls, ports, and communication errors**

- Solutions:
  - Relying less on network for communication
  - If necessary to communicate, a process must specify the IP address and port number across which to transmit this information
  - Backup Frequently
  - Better logging

#### Slow data transmission

- Solutions:
  - Avoid making RPC calls
  - Try higher bandwidth interconnects like NVLink and Infini-band
  - FP32 -> FP16 / Mixed precision
  - transmit a subset of gradients as soon as they are calculated (i.e. sending the gradients of a single layer) while at the same time, backpropagation is being performed on subsequent layers.

Reference: https://neptune.ai/blog/distributed-training-errors
