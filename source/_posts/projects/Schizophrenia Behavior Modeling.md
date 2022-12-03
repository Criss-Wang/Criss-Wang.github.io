---
date: 2022-07-31
layout: post
title: "Schizophrenia Behavior Modeling"
categories:
  - Projects
excerpt: "Relapse prediction using AI model pipeline"
mathjax: true
toc: true
---

### **Introduction**
Schizophrenia Behavior Modeling is a research and engineering project I worked on at [IMS-NUS](https://ims.nus.edu.sg/) in collaboration with [MOHT Singapore](https://www.moht.com.sg). In this project, I teamed up with a group of 4 people to develop enterprise-level medical tools for schizophrenia relapse prediction with AI models. Because of NDA, I cannot make the repo open source, but here is a glimpse of our work.

<figure align="center">
    <img src="/../../images/Projects/HOPES1.png" width="500px">
    <img src="/../../images/Projects/RIPS.png" width="580px">
</figure>

- Major contributions: 
    - Clinical scale regression model
        - **LSTM with recurrent dropout and layer normalization**
        <figure>
            <img src="/../../images/Projects/lstm.png" width="300px">
            <img src="/../../images/Projects/layer_norm.png" width="330px">
        </figure>
        - **1D ResNet** 
        <figure>
            <img src="/../../images/Projects/cnn.png" width="300px">
        </figure>
  	- **XGBoost-based** relapse classification model
    - Deep learning model pipeline for data parsing/transformation & model training & result feeding


### **Tech & Methodology**
<div>
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" width="40" height="40"/>&nbsp;
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pytorch/pytorch-original.svg"  width="40" height="40"/>&nbsp;
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/jupyter/jupyter-original.svg" width="40" height="40"/>&nbsp;
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/numpy/numpy-original.svg" width="40" height="40"/>&nbsp;
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pandas/pandas-original.svg"  width="40" height="40"/>&nbsp;
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/docker/docker-original.svg" width="40" height="40"/>&nbsp;
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/amazonwebservices/amazonwebservices-plain-wordmark.svg" width="40" height="40"/>&nbsp;
</div>
