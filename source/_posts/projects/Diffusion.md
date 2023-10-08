---
date: 2023-03-04
update: 2023-04-12
layout: post
title: "Motion Prediction with Guided Diffusion"
categories:
  - Projects
  - Diffusion
  - Deep Learning
  - Autonomous Vehicle
excerpt: "A D3.js-powered Satellite Tracking system"
link: "https://cdn.statically.io/gh/Criss-Wang/image-host@master/Blog/Diffusion1.webp"
mathjax: true
toc: true
---

### **Introduction**

We proposed a guided diffusion based method for Motion Forecasting task. The diffusion pro-
cess uses the standard UNet architecture with 1D-convolution conditioned on past locations. We addressed the problem of a long-tailed data distribution using a max-norm scaling. Our model outperformed baseline methods in experiments
using ArgoVerse 2 dataset.

#### Members

Andrew Shen, Zhenlin Wang, Yilong Qin

#### Resultant Prediction Plots

![](https://cdn.statically.io/gh/Criss-Wang/image-host@master/Blog/Diffusion1.webp)

[[**Code**](https://github.com/Criss-Wang/trajectory-diffusion)][[**Report**](https://drive.google.com/file/d/118t8mAokTr4-YEQ5pSTlT4QPrGgBUzN-/view?usp=drive_link)]
