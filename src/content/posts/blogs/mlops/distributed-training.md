---
title: "Understanding Distributed Training in Deep Learning"
excerpt: "What's important in industry AND research, not taught in school???"
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

Since last year, the quest for large X models have been nonstop, and people kept exploring the possibility to build more universal and robust models. While some still put a doubt if models with more parameters will be effective, most have faith in the [scaling law](https://arxiv.org/pdf/2001.08361.pdf) proposed by DeepMind and OpenAI researchers. The progress in 1 year is promising, as it seems that we are steadily moving towards the era of AGI. However, the education barely follows. College and Unversity are still bound by the budget to enable students to get in touch to large model training, especially when it comes to multi-gpu / multi-node distributed training. In light of this, I would love to share what I understand about distributed training, and how can we get started in this domain to catch up with recent industrial progress.

## 1. Definition

- Leverages multiple compute resources—often across multiple nodes or GPUs—simultaneously, accelerating the model training process.
- Mainly a form of parallelism, requires some understanding of low-level operation system (memory, communication and GPU architecture)
- For those interested, I will recommend taking [CMU 15-418 Parallel Computer Architecture and Programming](https://www.cs.cmu.edu/~15418/index.html) to get an in-depth understanding.

## 2. Parallelism in Training

- Two primary forms of parallelism: **model parallelism** and **data parallelism**

- **Model Parallelism**:
  - Used when a model doesn't fit into the memory of a single device.
  - Different parts of the model are placed on different devices, enabling the training process to occur across multiple GPUs or nodes. This approach is particularly useful for exceptionally large models.
- **Data Parallelism**:
  - Split the dataset across various devices, with each processing a unique subset of the data.
  - The model's parameters are then updated based on the collective gradients computed from these subsets (with different strategies).

## 3. Strategies in detail

**[Note]**: I\'ll mainly use PyTorch in this blog as it is the most popular and convenient choice. It is mainly based on `torch.distributed` package. In the meantime, some convenient scripts are created by _Lightning AI_ with their own libraries. I\'ll show some code using their library for people who just want a shortcut and get rid of the details behind distributed training.

1. Data Parallelism

   - How `DistributedDataParallel` works:
     - NCCL: multi-GPU, multi-node **communication** primitives. all-gather, all-reduce, broadcast, reduce-scatter, reduce routines, point-to-point send/receive. High bandwidth, low latency on **PCIe** and **NVLink** interconnects
     - All GPUs share same initial weights. Aggregate all gradients in different GPUs and update the weight collectively.
     - Need to update optimizer state and weights after AllReduce.
   - DDP Implementation

   ```python
   ### DDP - PyTorch Version
   import torch
   import torch.distributed as dist
   from torch.nn.parallel import DistributedDataParallel as DDP
   from torch.utils.data.distributed import DistributedSampler

   def main():
      # Initialize distributed environment
      dist.init_process_group(backend='nccl')

      # Create model
      model = YourModel()
      model = DDP(model)

      # Load data and distribute it across processes
      train_loader = DistributedSampler(YourDataset())

      # Training loop
      for epoch in range(epochs):
         for data in train_loader:
               inputs, labels = data
               outputs = model(inputs)
               loss = YourLoss(outputs, labels)

               # Backward and optimize
               optimizer.zero_grad()
               loss.backward()
               optimizer.step()
   ```

   - For more advanced details like **RPC-Based Distributed Training (RPC)** and **Collective Communication (c10d)**, refer to `torch.distributed` [original docs](https://pytorch.org/tutorials/beginner/dist_overview.html)
   - Fully Sharded DP (FSDP)

     1. What is in the GPU memory (x params, FP16)
        1. Params: 2x (fp16 with 2 bytes)
        2. Gradients: 2x
        3. Optimizer (AdamW)
           - Param copy: 4x (float32)
           - Momentum: 4x
           - Variance: 4x
     2. How FSDP works
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
     3. Reason to use/not to use FSDP
        1. When to use
           - Model size is too large (not data size)
           - More communication between GPUs
           - Hence trade memory for speed: more GPU memory cost due to communication, however, communication overhead reduced via NCCL acceleration
           - If want to trade speed for memory, see **activation checkpointing**
        1. When not to use
           - For models < 100 million params, consider activation-checkpointing and reversible layers
           - Recommend to use BFloat16 instead of Float16 (Float16 requires ShardedGradScaler)
           - Mixed Precision Training Concern (Package compatibility)
     4. FSDP Implementation

     ```python
     ### FSDP Version
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

2. Model Parallelism
   - split horizontally
   - Implementation
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
   - Inefficient sometimes: in the code above, GPU may be idle if layer 2 is not run during training
   - Does not work well if the model architecture does not naturally lend itself to being divided into parallelizable segments.
3. Pipeline Parallelism
   ![](https://www.mishalaskin.com/_next/image?url=%2Fimages%2Fpipeline_parallel.png&w=3840&q=75)

   - Mixed data and model parallelism, involves scheduling of data flow
   - Split into multiple stages, and each stage is assigned to a different device
   - The output of one stage is fed as input to the next stage.
   - Sometimes inefficient and suffers from idle time when machines are waiting for other machines to finish their stages: pipeline is waiting for a stage to finish in both the forward and backward pass, the period when some machine are idle aer referred to as a _bubble_.

4. Tensor parallelism

   - Split vertically + horizontally (in units of a tensor)
   - Can be more effective as it leverages efficiencies within matrix multiplication by spliting a tensor up into smaller fractions and expedite the computation
   - The detail can be expanded into another blog, however, I will refer you to this [excellent blog](https://www.mishalaskin.com/posts/tensor_parallel) instead of reinventing the wheel myself again.
   - Might require models specifically designed to take advantage of this form of parallelism. It may not be as universally applicable as data or model parallelism.

5. `torchrun`

   - An elegant way to run distributed training using `torch.distributed` package. Please refer to details [here](https://pytorch.org/docs/stable/elastic/run.html)
   - Make use of rendezvous backend to achieve high availability and failure recovery
   - A few major advantages include:
     - Single-node multi-worker
     - Multi-node
     - Multi-GPU
     - Fault tolerant
     - Elastic

6. Distributed Training on the Cloud

   - Since most of the resources are available from the cloud, and they are on-demand, it is common practice to migrate local code to be run on remote servers. You can spin up GPU resources (usually more capable than your local version) yourself and manage the dependencies/monitoring independenly, or you can resort to integrated solutions like AWS SageMaker or Azure ML or Google AI Studio as they often provide convenient API endpoints to interact with those GPU instances. In many scenarios, their management include inter-gpu/inter-node communication as well, which is a big plus.
   - As an example, you can setup AWS accordingly and run your distributed training using SageMaker via [this tutorial](https://docs.aws.amazon.com/sagemaker/latest/dg/distributed-training.html)
   - A sample script is as follows:

   ```python
   from sagemaker.pytorch import PyTorch

   estimator = PyTorch(
      ...,
      instance_count=2,
      instance_type="ml.p4d.24xlarge",
      # Activate distributed training with SMDDP
      distribution={ "pytorchddp": { "enabled": True } }  # mpirun, activates SMDDP AllReduce OR AllGather
      # distribution={ "torch_distributed": { "enabled": True } }  # torchrun, activates SMDDP AllGather
      # distribution={ "smdistributed": { "dataparallel": { "enabled": True } } }  # mpirun, activates SMDDP AllReduce OR AllGather
   )
   ```

7. Other packages

   - PyTorch Lightning - a lightweight PyTorch wrapper that provides a high-level interface for researchers and practitioners to streamline the training of deep learning models. It abstracts away many of the boilerplate code components traditionally required for training models, making the code cleaner, more modular, and more readable. It requires little setup of code and just need to insert a few parameters to the trainer
     - Example
     ```python
     trainer = L.Trainer(
        max_epochs=3,
        callbacks=callbacks,
        accelerator="gpu",
        devices=4,  # <-- NEW
        strategy="ddp",  # <-- NEW
        precision="16",
        logger=logger,
        log_every_n_steps=10,
        deterministic=True,
     )
     ```
   - Hugging Face `Accelerate`: a library that enables the same PyTorch code to be run across any distributed configuration by adding just four lines of code. It is still built on `torch_xla` and `torch.distributed`, but have get users rid of writing custom code to adapt to these platforms.

     - Benefits include easy utilization of ZeRO Optimizer from DeepSpeed, achieve FSDP and mixed-precision training as well.
     - Example

     ```python
     from accelerate import Accelerator
     accelerator = Accelerator()

     model, optimizer, training_dataloader, scheduler = accelerator.prepare(
         model, optimizer, training_dataloader, scheduler
     )

     for batch in training_dataloader:
           optimizer.zero_grad()
           inputs, targets = batch
           inputs = inputs.to(device)
           targets = targets.to(device)
           outputs = model(inputs)
           loss = loss_function(outputs, targets)
           accelerator.backward(loss)
           optimizer.step()
           scheduler.step()
     ```

     - In terminal, run `accelerate launch {my_script.py}`

## 4. Challenges and Solutions

1. Communication Overhead:

- In distributed training, the exchange of information between devices becomes a potential bottleneck. As the number of devices increases, coordinating updates and sharing gradients become more complex.
- Solutions:

  - Optimized Communication Protocols: Leveraging optimized communication protocols, such as NVIDIA NCCL for GPU communication, helps minimize the latency associated with inter-device communication.

  - Gradient Accumulation: By accumulating gradients locally on each device before synchronization, communication frequency is reduced. This strategy can be beneficial in scenarios where frequent synchronization is not necessary.

2. Fault Tolerance:

- In distributed environments, hardware failures or network issues are inevitable. Ensuring fault tolerance is essential to maintain the integrity of the training process.
- Solutions

  - Checkpointing: Regularly saving model checkpoints allows training to resume from the most recent checkpoint in case of a failure. This practice minimizes data loss and ensures continuity.

  - Redundancy: Introducing redundancy by running multiple instances of the training job across different nodes adds a layer of resilience. Load balancing techniques can be employed to distribute tasks effectively.

3. Scaling Issues:

- Scaling distributed training to a large number of nodes presents challenges in terms of efficiency and resource management.
- Strategies
  - Dynamic Resource Allocation: Implementing dynamic resource allocation ensures that resources are allocated efficiently based on the current load. Kubernetes and other orchestration tools can facilitate dynamic scaling.
  - Parameter Servers: Utilizing parameter servers, which are dedicated servers responsible for storing and distributing model parameters, can enhance the scalability of distributed training.

## 5. Common Mistakes

1. Not pipelining

- Pipeline Parallelism is always something to include. Notice the use of ZeRO-3 also uses pipeline parallelism

2. Not balancing pipeline stages

- There will be some brief periods where either a machine is idle and waiting on the next minibatch from the previous machine or takes longer than other machines to execute its computation, thus slowing down the pipeline.
- You should ideally construct your pipeline such that each machine does as close to the same amount of computation as possible. This means timing how long it takes data to get through different layers in the model, timing how long forward and backward propagation takes for each model partition, and ensuring roughly equivalent data sizes across mini-batches. This is critical for optimizing pipeline efficiency.
- To achieve this, setting up profiler like PyTorch Profiler is critical for evaluation of computations done during model training

3. Weight staleness

- When model training is pipelined across multiple machines, there is a delay that happens between when the forward computation on data occurs and when the gradients based on that computation are backpropagated to update the model weights. As a result, forward propagation are calculated using weights that aren\'t updated with the latest gradients.
- Solution: **weight stashing**
  A system “maintains multiple versions of a model’s weights, one for each minibatch.” After the completion of each forward pass, the system can store a model’s weights as part of the state associated with that minibatch. When the time comes for backpropagation, the weights associated with that minibatch are retrieved from the stash and used for the gradient computation. This ensures that the same version of weights are used for the forward and backward pass over a single minibatch of data within a pipelined stage, and statistical convergence is improved.

4. Driver and library inconsistencies between machines

- Containerization / Virtualization using tools like Docker solves the problem

5. Wrong type of Optimizer Update

- Example: Synchronous vs Asynchronous SGD
- Asynchronous SGD ([HogWild](https://people.eecs.berkeley.edu/~brecht/papers/hogwildTR.pdf) as a popular choice) which showed that SGD could be run in parallel, without locks, and without too much effect on algorithm convergence. Asynchronous SGD allows weight updates to proceed without each machine waiting for the other to send their gradients.

6. Network issues, firewalls, ports, and communication errors

- Solutions:
  - Relying less on network for communication
  - If necessary to communicate, a process must specify the IP address and port number across which to transmit this information
  - Backup Frequently
  - Better logging

7. Slow data transmission

- Solutions:
  - Avoid making RPC calls
  - Try higher bandwidth interconnects like NVLink and Infini-band
  - FP32 -> FP16 / Mixed precision
  - transmit a subset of gradients as soon as they are calculated (i.e. sending the gradients of a single layer) while at the same time, backpropagation is being performed on subsequent layers.

## 6. A complete Distributed DL pipeline

1. Distributed Training Setup:

   - Set up a distributed computing environment, typically using a cluster or cloud infrastructure like AWS, Google Cloud, or Azure.
   - Ensure that all nodes in the cluster have the necessary libraries (TensorFlow, PyTorch, etc.) and dependencies installed.
   - Split the training dataset across nodes to distribute the workload.

2. Synchronization and Communication:

   - Implement a synchronization mechanism to ensure that the model's weights are updated consistently across all nodes.
   - Choose a communication protocol (e.g., Parameter Server, AllReduce) for aggregating gradients and exchanging model updates.

3. Model Initialization:

   - Initialize the same model architecture with random weights on all nodes.
   - Setup model to follow data parallelism

4. Training Loop (The main discussion we had in the blog):

   - Start the training loop on each node with its batch of data.
   - Compute gradients for the batch, update local weights, and synchronize with other nodes.
   - Repeat this process for a predefined number of epochs.
   - Implement early stopping to prevent overfitting and save the best-performing model checkpoint during training.
   - Periodically evaluate the model's performance on the validation dataset to ensure it's learning effectively.
   - Save model checkpoints at regular intervals during training to resume from a specific point in case of interruptions.
   - If necessary, scale up the distributed training environment by adding more nodes to further accelerate training or handle larger datasets.

5. Monitoring and Logging:

   - Implement monitoring and logging to track training progress, including loss, accuracy, and other relevant metrics.
   - Use tools like TensorBoard or custom logging solutions to visualize training statistics.

6. Hyperparameter Tuning:

   - Perform hyperparameter tuning, which may include learning rate adjustments, batch sizes, and other parameters, to optimize the model's performance.
   - **Note**: you should set a budget alert before this, as running multiple experiments (on a large model) in a distributed setting can be very COSTLY!!!

7. Post-training Analysis:

   - This can go before/after/hand-in-hand with step 6, as part of model tuning
   - Analyze the trained model's performance on the test dataset to assess its generalization capabilities.

8. Deployment:

   - Deploy the trained model for inference in your production environment, whether it's on the cloud or at the edge.
   - Sometime this requires distributing model weights across servers as well

9. Additional Fine-tuning (Optional):

   - Fine-tune the model as needed based on deployment feedback or new data.
   - Checkout Hugging Face's [TRL library](https://huggingface.co/docs/trl/en/index) & its tutorials to understand more.

10. Documentation:

- Document the entire distributed training process, including configuration settings, data preprocessing steps, and model architecture, for future reference.

11. Maintenance and Updates:

- Regularly update and maintain the distributed training system, including libraries, dependencies, and data pipelines, to ensure its reliability and performance.

For the basic scripts without distributed training and with basic DDP. You may refer to the [tutorial here](https://theaisummer.com/distributed-training-pytorch/). If you want a one-off solution, please refer to the code below.

## 7. A more challenging code using native PyTorch

If you are interested in building it from scratch with PyTorch directly, checkout the following code (if you don't understand the syntax, please DIY)

```python
"""A demo on how to setup custom trainer with efficient training"""
import os
import argparse
import apex.amp as amp
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

class ConvNet(nn.Module):
   def __init__(self, num_classes=10):
      super(ConvNet, self).__init__()
      self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
      self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
      self.fc = nn.Linear(7*7*32, num_classes)

   def forward(self, x):
      out = self.layer1(x)
      out = self.layer2(out)
      out = out.reshape(out.size(0), -1)
      out = self.fc(out)
      return out

def ddp_setup()  #(rank: int, world_size: int):
   """
   Args:
      rank: Unique ID of each
      world_size: Total number of processes
   """
   # multi-gpu setup
   # os.environ['MASTER_ADDR'] = 'your master node ip'
   # os.environ['MASTER_PORT'] = '8888'
   # dist.init_process_group(
   #     backend='nccl',
   #     init_method='env://',
   #     world_size=world_size,
   #     rank=rank
   # )
   dist.init_process_group(backend="nccl")

class Trainer:
   def __init__(
      self,
      model: torch.nn.Module,
      train_data: DataLoader,
      optimizer: torch.optim.Optimizer,
      criterion: torch.nn.Module,
      # gpu_id: int,
      save_every: int,
      snapshot_path: str,
   ) -> None:
      # self.gpu_id = gpu_id
      self.local_rank = int(os.environ["LOCAL_RANK"])
      self.global_rank = int(os.environ["RANK"])
      self.model = model.to(self.local_rank)
      self.train_data = train_data
      self.optimizer = optimizer
      self.criterion = criterion
      self.epochs_run = 0
      self.save_every = save_every

      self.model, self.optimizer = amp.initialize(
            self.model, self.optimizer, opt_level='O1')
      if os.path.exists(snapshot_path):
            print("Loading Snapshot")
            self._load_snapshot(snapshot_path)
      self.model = DDP(self.model, device_ids=[self.local_rank])

   def _load_snapshot(self, snapshot_path):
      """Resume training from previous checkpoint"""
      snapshot = torch.load(snapshot_path)
      self.model.load_state_dict(snapshot['model'])
      self.optimizer.load_state_dict(snapshot['optimizer'])
      self.epochs_run = snapshot["epochs_resume"]
      amp.load_state_dict(snapshot['amp'])
      print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

   def _run_batch(self, source, targets):
      self.optimizer.zero_grad()
      output = self.model(source)
      loss = self.criterion(output, targets)
      loss.backward()
      with amp.scale_loss(loss, self.optimizer) as scaled_loss:
            scaled_loss.backward()
      self.optimizer.step()
      return loss.item()

   def _run_epoch(self, epoch: int, total_epochs: int):
      self.model.train()
      for i, (source, targets) in enumerate(self.train_data):
            source = source.to(self.local_rank)
            targets = targets.to(self.local_rank)
            loss = self._run_batch(source, targets)
            if (i + 1) % 100 == 0 and self.local_rank == 0:
               print(
                  f'[GPU{self.global_rank}] Epoch [{epoch + 1}/{total_epochs}], Step [{i + 1}/{len(self.train_data)}], Loss: {loss:.4f}')

      self.model.eval()
      with torch.no_grad():
            for i, (source, targets) in enumerate(self.val_data):
               source = source.
               targets = targets.
               loss = ...

   def _save_snapshot(self, save_dir, epoch, model_seed):
      path = f"{save_dir}/base_model_seed={model_seed}_epoch={epoch}.pt"
      torch.save({
            'model': self.model,  # if saving state_dict, use .module.state_dict()
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'amp': amp.state_dict(),
            'epochs_resume': self.epochs_run
      }, path)
      print(f"Epoch {epoch} | Training snapshot saved at {path}")

   def train(self, total_epochs: int, model_seed: int, save_dir: str):
      """Training script"""
      for epoch in range(self.epochs_run, total_epochs):
            self._run_epoch(epoch, total_epochs)
            if self.local_rank == 0 and epoch % self.save_every == 0:
               self._save_snapshot(save_dir, epoch, model_seed)


def load_train_params():
   train_set = MyTrainDataset(2048)
   model = ConvNet()
   optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
   criterion = torch.nn.CrossEntropyLoss()
   return train_set, model, optimizer, criterion


def prepare_dataloader(dataset: Dataset, batch_size: int, num_workers: int, sampler: DistributedSampler):
   return DataLoader(
      dataset,
      batch_size=batch_size,
      pin_memory=True,
      shuffle=True,
      num_workers=num_workers,
      sampler=sampler
   )


def run(args):
   """Run entire pipeline"""
   torch.manual_seed(args.seed)
   # rank = args.nr * args.gpus + gpu
   # ddp_setup(rank, args.world_size)
   ddp_setup()
   dataset, model, optimizer, criterion = load_train_params()
   # sampler = DistributedSampler(
   #     dataset, num_replicas=args.world_size, rank=rank)
   sampler = DistributedSampler(dataset)
   train_data = prepare_dataloader(
      dataset, batch_size=32, num_workers=0, sampler=sampler)
   trainer = Trainer(model, train_data, optimizer,
                     criterion, args.save_every, args.snapshot_path)
   trainer.train(args.total_epochs, args.seed, args.save_dir)
   dist.destroy_process_group()


def main():
   """Entry point

   to call the script using torchrun (which manages the )

   e.g: 4 GPU per machine, 50 epochs, saving every 10 epoch
   torchrun \
   --nproc_per_node=4 \
   --nnodes=2 \
   --node_rank=0 \
   --rdzv_id=456 \
   --rdzv_backend=c10d \
   --rdzv_endpoint=HOST_MACHINE:PORT \
   FILE_NAME.py --epochs=50 --save_every=10

   nproc_per_node: usually num of GPUs per machine
   nnodes: num of machines
   node_rank: ID: 0/1/2/.... for each machine
   Notes on endpoint: choose endpoint whose machine has high network bandwidth
   Note: Multi-GPU on single machine is much faster than Multi-node each with single GPU due to bandwidth limit over TCP
   """
   parser = argparse.ArgumentParser()
   # parser.add_argument('-n', '--nodes', default=1,
   #                     type=int, metavar='N')
   # parser.add_argument('-g', '--gpus', default=1, type=int,
   #                     help='number of gpus per node')
   # parser.add_argument('-nr', '--nr', default=0, type=int,
   #                     help='ranking within the nodes')
   parser.add_argument('--epochs', default=2, type=int,
                        metavar='N',
                        help='number of total epochs to run')
   parser.add_argument('-s', '--seed', default=42,
                        type=int, metavar='N')
   parser.add_argument('--save_every', default=5, type=int,
                        help='interval to save the snapshot')
   parser.add_argument('--save_dir', default='.', type=str,
                        help='directory to save the snapshot')
   parser.add_argument('--snapshot_path', default='.', type=str,
                        help='path of the snapshot to restore training from')
   args = parser.parse_args()
   #########################################################
   args.world_size = args.gpus * \
      args.nodes if args.gpus >= 0 else torch.cuda.device_count()
   args.total_epochs = args.epochs

   # mp.spawn(main, nprocs=args.world_size, args=(args,))
   run(args)
   #########################################################



########## POST MORTEM ###################
"""
Common Troubleshooting
1. Check nodes can communicate through **TCP**
2. Check inbound rules on a security group (on AWS)
3. export NCCL_DEBUG=INFO to set verbose logs
4. export NCCL_SOCKET_IFNAME to ensure TCP connection is correct

SLURM work scheduler setup TODO

"""
```

## References

- https://neptune.ai/blog/distributed-training-errors
