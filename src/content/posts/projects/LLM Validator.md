---
date: 2024-09-20
updated: 2025-10-05
layout: post
title: "LLM Validator"
categories:
  - Projects
  - Python
  - LLM Evaluation
  - Benchmarking
  - MLOps
excerpt: "A configurable LLM benchmarking template for repeatable model, prompt, dataset, and metric validation"
link: "/images/Projects/llm-validator-diagram.svg"
mathjax: true
toc: true
---

### Introduction

[LLM Validator](https://github.com/Criss-Wang/llm-validator) is a validation pipeline template for comparing language models, prompts, datasets, and metrics without turning every experiment into a one-off notebook.

The project supports provider-specific clients, custom prompts, benchmark datasets, and configurable metrics across cost, latency, accuracy, security, and stability. It is meant to make model changes easier to rerun and easier to audit when a prompt, dataset, or model provider changes.

### What it does

- Defines prompt and dataset inputs as project files
- Runs repeatable model validation from JSON configs
- Supports custom inference clients and local endpoints
- Tracks model quality with configurable metrics
- Pairs with the model-validation writeup on this site

### Related writeup

The project is used in my [model validation post](/writing/software/model-iteration-research-validation/), where I walk through how validation infrastructure can make model iteration less anecdotal and more reproducible.
