---
date: 2023-07-29
layout: post
title: "Needle: High-performance DL System"
categories:
  - Projects
  - Deep Learning
  - System Programming
  - CUDA Programming
excerpt: "A Deep Learning framework with customized GPU and CPU backend in C++ and Python"
link: "https://cdn.statically.io/gh/Criss-Wang/image-host@master/Blog/Needle2.50se01grtxw0.webp"
mathjax: true
toc: true
---
### Introduction

Needle is a Deep Learning framework with customized GPU and CPU backend in C++ and Python. This is an attempt to simulate PyTorch\'s **imperative** style, especially its way of auto-differentiation and computational graph traversal. In the meantime, we enable **accelerated computing** with custom *ndarrays* implementation via low level C++ CUDA programming. This enables tensor operations to run on GPUs and other specialized hardwares.

### Key contributions

- Modular DL framework
  - ![](https://cdn.statically.io/gh/Criss-Wang/image-host@master/Blog/Needle2.50se01grtxw0.webp)
- Build models from scratch
  - ResNet (Residual Blocks, Skip Connection, BatchNorm2D)
  - LSTM (Cell State, Hidden State, Forget/Input/Output Gate, Activations)
  - Transformer (Multi-Head Self/Cross Attention, LayerNorm, Dropout, Positional Encoding, FFN, Skip Connection, Attention Masking)
- Optimization with GPU & CUDA
  - SIMT
  - Tensor Model Parallelism
  - Register Tiling
  - Block Tiling
  - GPipe
  - Best result: ~5x speedup in 4-core distributed training

### **Tech Stack & Methodology**

<div>
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" width="40" height="40"/>&nbsp;
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/cplusplus/cplusplus-original.svg"  width="40" height="40"/>&nbsp;
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/cmake/cmake-original.svg" width="40" height="40"/>&nbsp;
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/bash/bash-plain.svg"  width="40" height="40"/>&nbsp;
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linux/linux-original.svg" width="40" height="40"/>&nbsp;
</div>

### Acknowledgement

This project is inspired by [10-414/714](https://dlsyscourse.org) *Deep Learning Systems* by [Carnegie Mellon University](https://www.cmu.edu). Extensions based on this are built and are still under development (more to come!).
