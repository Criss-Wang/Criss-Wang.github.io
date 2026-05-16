---
date: 2024-06-23
updated: 2024-08-30
layout: post
title: "Deployable AI"
categories:
  - Projects
  - Python
  - Model Serving
  - FastAPI
  - MLOps
excerpt: "A lightweight local inference-serving toolkit for registering models and exposing prediction endpoints quickly"
link: "/images/Projects/dpai-serving-diagram.svg"
mathjax: true
toc: true
---

### Introduction

[Deployable AI](https://github.com/Criss-Wang/dpai) is a small toolkit for serving local model inference quickly. It follows a familiar model-package pattern: register a model artifact, provide an inference script, and expose the model through a predictable API endpoint.

### Workflow

The repo centers on three pieces:

- A serialized model artifact, usually saved as a `.joblib` file
- An inference script with `input_fn` and `predict_fn`
- A serving command that starts a local backend with one endpoint per registered model

### Notes

The project is intentionally narrow: it targets JSON request workflows and local development speed. It is useful as a compact serving layer when the goal is to validate model behavior before investing in heavier deployment infrastructure.
