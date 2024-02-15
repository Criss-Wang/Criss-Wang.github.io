---
title: "DL Training: An in-depth discussion"
excerpt: "Training can be hard a lot of time. But the bottlenecks vary"
date: 2024/02/19
categories:
  - Blogs
tags:
  - Deep Learning
  - MLOps
layout: post
mathjax: true
toc: true
---

## Introduction

Training in the field of Deep Learning has been a well-studied component which contributed significantly to the development of large-scale models. As a pivotal part of model research, optimizing training of a model can lead of numerous benefits including cost saving, energy consumption reduction and expedition of effective model discovery. Personally I feel that it can be taught as an actual coursework due to the load of content it has and the fact that every DL researcher and engineer should master training thoroughly to be truly productive. Unfortunately, no school or even online coursework has considered this, partly because that the overhead cost of leveraging multi-gpu or gpu cluster can be high. Nonetheless, I\'m optimistic that as we further scale the gpu resources and advent our cloud computing technology, my proposal will soon become feasible. But before that happens, let\'s have a simplified "course" on it in this blog post.

## Assumptions

This blog assumes the mastery of basic training pipeline and will only focus on advanced training techniques. We would also assume that data is clean and preprocessed. So anything related to data quality, feature transformation, tokenization or data loader definitions will be out of the scope of this blog. Instead, we focus on a training of a single (usually large scale) model with a given set of hyperparameters. In the meantime, we also have a fixed loss/evaluation metric for a given task during training. To learn more about these components. Checkout my post on [designing a large-scale deep learning system](http://www.criss-wang.com/post/blogs/deep-learning-system-design-1/)

## Optimizers

[TODO] Read optimization chapter of Ian\'s book

Very often, one would also need to consider the scheduling of learning rate. The purpose is to prevent the model from converging too quickly and getting stuck in local minima. In the case of large-scale models, this is less of a problem. As studies have shown, the proximity of local minima to the global answer is generally good when model size is big enough. However, it has also been shown that warming up the gradients and rate scheduling will help stabilize training, leading to easier experiment monitoring as a consequence. Here are the overview of a few common schedulers.

1. **Inverse Square Root (ISR)**: this is one of the simplest and most common learning rate schedules, where the learning rate is set to 0 at the beginning of training and then increased by a fixed factor each epoch until reaching a maximum value, after which it remains constant for the rest of training. This helps prevent overfitting by gradually increasing the learning rate as the model becomes more efficient during early stages of training, while maintaining a steady pace throughout the remaining period.

2. **Exponential Decay (ED)** - this is another simple and popular learning rate schedule that involves using an exponential function to adjust the learning rate over time. The learning rate is set to 0 at the beginning of training and then increased exponentially until reaching a maximum value, after which it remains constant for the rest of training. The allows for faster descent at early stage and more refined exploration at a later stage

3. **Stepwise Schedule** - this is another simple and popular learning rate schedule that involves using discrete steps to adjust the learning rate over time. The learning rate is set to 0 at the beginning of training and then increased by a fixed factor each step until reaching a maximum value, after which it remains constant for the rest of training.

## Memory Reduction in Training

As DL model sizes grow bigger and bigger in recent years, our hardware (GPU, TPU) can hardly handle the amount of parameters in the model. Caching everything direclty in-memory is no longer a viable option, especially for individual researchers/engineers with limited support of GPU. Therefore, reducing memory consumption in training becomes extremely important. Using the following techniques, we can reduce the size of weights, biases and activations while maintaining accuracy, and hence enable smooth training without the heinous "Out-Of-Memory" error.

1. **Automatic Mixed Precision (AMP)**
   When working with large datasets or deep models where computing gradients on all data points can become computationally expensive, using a higher precision (e.g., floating-point) during training may result in slower convergence times due to the increased computational cost. On the other hand, using fixed-point arithmetic may not be able to capture fine details of the model\'s behavior and lead to lower accuracy.

   To address this trade-off, automatic mixed precision algorithms select a precision level based on various factors such as memory usage, computation time, and accuracy requirements. In general, floating-point is used during training when more precision is needed (e.g., large weights or small gradients) while fixed-point arithmetic is used for inference when lower precision can result in faster performance (e.g., smaller models).

2. **True Low-Precision Training**
   In contrary to AMP, true low-precision training is needed when size of parameters are still too big. This is one of the last resorts to sacrifice accuracy for training completeness. In this process, all parameters and gradients are in lower precision and are converted back to higher precision after training.

3. **Reduce Batch Size**
   One of the easiest strategies to adopt when OOM appears is to use a smaller batch size. By doing so, we have much less data saved in cache, as the matrix used to compute gradient is much smaller (a scale of O(N) where N is number of parameters). However, the risk of slower convergence, unstable training and lower accuracy is closely related to a small batch size as well. When mini-batch SGD is "too close" to SGD, the benefits brought multicore processing will also diminsh. So be careful when using it.

4. **Gradient Accumulation**
   When reducing batch size leads to severely poorer performance, and increasing it causes infeasible training, we have a work-around, gradient accumulation, which virtually increase the batch size during training. This is easily achievable (see following code):

   ```python
   for batch_idx, batch in enumerate(train_loader):
      model.train()

      # FORWARD AND BACK PROP
      outputs = model(
         batch["input_ids"],
         attention_mask=batch["attention_mask"],
         labels=batch["label"]
      )

      outputs["loss"] = outputs["loss"] / accumulation_steps # accumulate loss
      fabric.backward(outputs["loss"])  # optimizer accumulate the gradient derived from this loss

      ### UPDATE MODEL PARAMETERS
      if not batch_idx % accumulation_steps or (batch_idx + 1 == len(train_loader)):
         # update every accumulation_steps iterations
         optimizer.step()
         optimizer.zero_grad()
   ```

   Note that this method is at a cost of slower training due to the sequential update within the "larger" batch.

5. **Optimizer**

   Sometimes, choose a stateless optimizer like SGD instead of Adam/AdamW will also help reduce memory consumption. Optimizer states can take up a significant amount of memory. For example, Adam optimizer requires first+second-order derivatives to be stored in the state for future updates. This is a additional storage term, as you can see from [this image](https://docs.oneflow.org/en/master/cookies/imgs/Three_Stages_of_ZeRO-DP_Optimizations.jpg). On the other hand, SGD is simply stateless, and requires only an O(N) temporary storage of first-order gradients. Hence, you can consider switching from the commonly used Adam to SGD if it doens\'t result in significant performance drops.

6. **Distributed Training and Tensor Sharding**
   Perhaps the most widely used strategy in industry when it comes to memory optimization is the use of sharding and distributed training. Companies have the resources and compute to perform model training on multiple GPUs and distribute the tensors across multiple servers. This is closely tied to the concept of parallelism. There are many forms of parallelisms in distributed training:

   - Data Parallelism
   - Model Parallelism
   - Tensor Parallelism
   - Pipeline Parallelism

   Each of them aims to fix some inefficiencies in the training pipeline. Many packages have been developed to implement these strategies for the benefits of DL researchers and engineers. I provide a more in-depth discussion in a different [post](http://www.criss-wang.com/post/blogs/distributed-training), together with discussion about the common errors people face during distributing training.

7. **Gradient Checkpointing**
   
   Gradient Checkpointing is an ingenious method to reduce memory usage by repeated discard . It comes from the observation that tnhe most memory intensive part of training deep neural networks is computing the gradient of the loss by backpropagation. By checkpointing nodes in the computation graph defined by your model, and recomputing the parts of the graph in between those nodes during backpropagation, it is possible to calculate this gradient at reduced memory cost. THe selected checkpoint nodes in the computational graph are kept in memory after the forward pass, while the remaining nodes are recomputed at most once. After being recomputed, the non-checkpoint nodes are kept in memory until they are no longer required. It does slow down training, but the benefit is a reduction to square-root scale of memory.

   You may resort to PyTorch's `autograd` library to easily craft a simple Checkpoint feature (note how the `ctx` has enable saving of function and args from `forward` method to be applied in `backward` later):
   ```python
   import torch
   import torch.nn as nn
   import torch.autograd as autograd

   class CheckpointFunction(autograd.Function):
      @staticmethod
      def forward(ctx, func, *args):
         ctx.func = func
         ctx.args = args
         with torch.no_grad():
               ctx.save_for_backward(*args)
               result = func(*args)
         return result

      @staticmethod
      def backward(ctx, grad_output):
         args = ctx.saved_tensors
         func = ctx.func
         f_args = tuple(args)
         f_args += (grad_output,)
         with torch.enable_grad():
               grad_input = torch.autograd.grad(func(*f_args), f_args, allow_unused=True)
         return (None,) + grad_input[1:]

   class CheckpointModule(nn.Module):
      def __init__(self, module):
         super(CheckpointModule, self).__init__()
         self.module = module

      def forward(self, *args):
         return CheckpointFunction.apply(self.module, *args)

   # Define a simple feed-forward network
   class SimpleNet(nn.Module):
      def __init__(self):
         super(SimpleNet, self).__init__()
         self.fc1 = nn.Linear(1000, 500)
         self.fc2 = nn.Linear(500, 200)
         self.fc3 = nn.Linear(200, 10)

      def forward(self, x):
         x = torch.relu(self.fc1(x))
         x = torch.relu(self.fc2(x))
         x = self.fc3(x)
         return x

   # Wrap the network with checkpointing
   checkpointed_net = CheckpointModule(SimpleNet())

   # Example usage
   input_data = torch.randn(1, 1000, requires_grad=True)
   output = checkpointed_net(input_data)
   loss = output.sum()
   loss.backward()

   print("Gradients w.r.t input_data:", input_data.grad)

   ```

   A simple example using Hugging Face\'s `transformer` library looks like this:

   ```python
   trainer = Trainer(
      ...
      args=TrainingArguments(
         ...
         gradient_checkpointing=True,
      ),
      ...
   )
   ```

8. **Parameter offloading**
   While the model is training via gradient updates, it is usually a subsection of the parameters in the model that gets updated every time. This leads to idleness of GPU memory, which simply stores the result of the unused parameters before their turn of update. To further utilize this segment of the memory, people found a way to offload some parameters from GPU to CPU, and only reload it for update/computation whenever necessary. While this definitely increases communication cost and slows training down, it reduces memory usage by a large scale. In some other scenarios (e.g. ZeRO-3), optimizers states are also offloaded to save memory, which further reduces memory consumption.

9. **Flash Attention**
   To improve the performance of Transformer-based models, flash attention was introduced by researchers at Google and Stanford University as an alternative to traditional self-attention mechanisms, which can be computationally expensive and prone to vanishing gradients.

   In essence, Flash Attention works by using a small window size (e.g., 512) to capture local context within the input sequence. The model then applies two separate attention mechanisms: one that uses full-length self-attention to capture long-range dependencies, and another that uses Flash Attention to focus on shorter-range patterns.

   The idea is that by using a smaller window size for the latter mechanism, the model can achieve better performance while still capturing important local features of the input sequence. This allows for more efficient computation and improved performance in downstream tasks like text classification or machine translation.

10. **Model Distillation**
    Sometimes, it is just easier to reduce the model size when memory constraint is hit. This had long been a success aside from techiniques such as model pruning from the invention of Jeffery Hinton. Model Distillation, or knowledge distillation, transfers knowledge from a larger and more complex model to a smaller and simpler one. Intuitively, the larger model is trained on a given dataset. Then it will act as a "teacher" to generate output representations that can be interpreted by the smaller model. These output representations are often in the form of probability distributions over the original input data, which allows for easier interpretation and use in downstream tasks. On the other hand, the smaller "student" model will try to predict outputs based on these probability distributions. Distillation is an enormous field of study in DL, and people often use existing libraries and distilled models made available by thousands of generous researchers who willingly release their trained model weights to the public.

11. **Quantization**
    Another way to directly reduce model size is via quantization. Quantization typically involves mapping each value in the input to a range of possible output values, using techniques like thresholding or rounding. This is done by defining a set of bins that cover the entire range of possible outputs for a given input, and then assigning each output to its corresponding bin based on some threshold value.

    During training, the weights and activations are typically quantized before being stored on disk or transmitted over networks in order to reduce their size without sacrificing too much accuracy. Additionally, the gradients of the loss function can be computed using an approximate gradient method called stochastic gradient descent with quantization
    (SGD-Q), which allows for faster convergence and improved performance compared to standard SGD methods.

    Do note that this method is slightly different from the low-precision or mixed precision strategy, as the former often recovers the precision after training, but this method allows the model trained to stay in the same precision during inference, which effecitvely reduced persistent model storage requirement as well. To learn more about quantization, checkout [this blog](http://www.criss-wang.com/post/blogs/quantization)

    A sample code implementation looks like this:

    ```python
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training, get_peft_model

    model_name = 'gpt2-large'
    quantization_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_use_double_quant=True,
      bnb_4bit_compute_dtype=torch.bfloat16,
    )
    # load model
    model = AutoModelForCausalLM.from_pretrained(
         model_name,
         device_map="auto",
         quantization_config=quantization_config,
         trust_remote_code=True,
    )
    ```

    Note that if you are using `peft` (e.g. for LoRA), you should also consider a quantized version for it using `prepare_model_for_kbit_training`:

    ```python
    ...
    from peft import prepare_model_for_kbit_training
    # preprocess the quantized model for traininng
    model = prepare_model_for_kbit_training(model)
    ```

## Using PyTorch Lightning for simplified training

Here we use lightning as an example to show how to quickly scale up training using the points mentioned above.

1. Lightning Module
   - `ModelCheckpoint`, `CSVLogger`
2. Automatic Mixed Precision Training

```python
import Lightning as L
class LightningModel(L.LightningModule):
	...

model = SomeModelLoader.from_pretrained(...)
lightning_model = LightningModel(model)

callbacks = [
	ModelCheckpoint(save_top_k=1, mode="max", monitor="val_acc")  # save top 1 model
]
logger = CSVLogger(save_dir="logs/", name="my-model")
trainer = L.Trainer(
	max_epochs=3,
	callbacks=callbacks,
	accelerator="gpu",
	precision="16",  # <-- NEW
	devices=[1],
	logger=logger,
	log_every_n_steps=10,
	deterministic=True,
)
```

3. Static Graphs with `Torch.Compile`
   - New feature in Torch 2.0
   - speed up PyTorch code execution by generating optimized static graphs instead of running PyTorch code with dynamic graphs
   - Under the hood, this is a 3-step process including graph acquisition, graph lowering, and graph compilation. [see official explanation](https://pytorch.org/get-started/pytorch-2.0/#pytorch-2x-faster-more-pythonic-and-as-dynamic-as-ever)
   - May not speed up necessarily (use it when dominated by model runtime rather than graph computation) -> initial optimization compilation step takes a few minutes but eventually accelerates the model training
   - Code may break down in distributed settings

```python
...
model = SomeModelLoader.from_pretrained(...)
model = torch.compile(model)
lightning_model = LightningModel(model)
...
```

4.  DDP (model + data parallel -> pipeline parallel)

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

5. Sharding
   - `strategy="deepspeed_stage_2"` to replace `strategy="ddp"`
   - note that `deepspeed_stage_2` refers to the stage two of [Zero Redundancy Optimizer (ZeRO)](https://docs.oneflow.org/en/master/cookies/zero.html#:~:text=Specifically%2C%20ZeRO%2DDP%20has%20three,of%20traffic%20as%20data%20parallel.), which effectively achieves the tensor sharding for optimizer states and gradients.
   - note that here stages mean different sharding strategies. Consider optimizer states/gradients/weights to be sharded and optimizer state to be offloaded to CPU (refer to Paged Optimizers if can\'t remember) if necessary. This corresponds to different stages of ZeRO method as well.
6. Use Lightning `Fabric` module (faster than `Trainer`). You may look it up online for further references.

## A more challenging code using native PyTorch

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

## Manage your training process

After all these training optimizations, it would be wasteful not to save your experiment data. Therefore, post-training processing becomes extremely important. Here is a brief list of actions to take note of:

- Metrics Monitoring: keeping the experiment data in persistent storage like MongoDB or SQL DB via mature mlops platforms like _MLflow_, _Weights and Biases_ or _Kubeflow_ is a convenient and important step to further analyse progress in model training and debugging potential bottlenecks/errors.
- Performance Monitoring: finding inefficiencies in hardward utilization (idle processes, slow hardware communication, low memory usage) can help further speed up your training and saving you tons of money. Your choice of server communication tools become important in this sense as well, and monitoring that part can be very tricky to carry out if you don\'t have sufficient low-level knowledge.
- Logging: error logging is not an easy thing when it comes to model training. Problems often can only be found from a parameter/matrix level, and the fact that training is distributed accross servers make it even harder to achieve.
- Model Checkpointing: saving you model half way during the training is critical to achieve fault-tolerant training. PyTorch maintainers has developed a thorough set of tools like `dcp`, `elastic`, `rendezvous` and `torchrun` in collaboration with deepspeed to help achieve that.

If you are really into completing the full training cycle and squeezing every last sip of the resources you\'ve paid for, I would urge you to check out [my other post](https://criss-wang.com/post/blogs/ml-post-training) to learn more about it.

## To conclude...

Training a model involves significant amount of design decisions that vary based on model, domain and data. It requires many years of experiences and mistake-making to form the right intuitions for model training setups. That said, constantly practicing it using some personal projects is what I did, and what I would recommend every passionate ML researcher/engineer should do to keep themselves updated with the latest progress in DL training. It might not "make perfect", but it certainly saves you and your company enormous time and money in the day-to-day interactions with every-bloating monsterous models.

## References and Further Readings

1. https://huggingface.co/docs/text-generation-inference/conceptual/flash_attention
2. https://sebastianraschka.com/blog/2023/pytorch-faster.html
