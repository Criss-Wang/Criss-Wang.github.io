---
title: "Writing Quality Code for Machine Learning"
date: 2024-03-09
categories:
  - Blogs
  - Draft Notes
tags:
  - MLOps
  - Code Quality
excerpt: "Practical notes on turning ML code from proof-of-concept scripts into maintainable systems."
layout: post
---

## Introduction

Machine learning code often begins as an experiment: a notebook, a script, a copied preprocessing block, a training loop, and a quick evaluation. That is fine at the proof-of-concept stage. The problem starts when experimental code quietly becomes production code without changing shape.

Quality ML code should make the important parts easy to change:

- Data loading and validation
- Feature transformation
- Training and evaluation
- Inference and serving
- Configuration
- Monitoring and rollback

The goal is not to make every project enterprise-heavy. The goal is to prevent the codebase from becoming impossible to test, extend, or debug once the model matters.

## 1. Proof-of-Concept Code Becomes Permanent

**Issue:** Proof-of-concept code is usually optimized for speed, not maintainability. Five to ten separate scripts may repeat the same preprocessing, feature engineering, training, deployment, and monitoring logic.

**Why it happens:** Early ML work rewards fast iteration. In startups and research-heavy teams, the fastest path is often to copy a working script and modify it. That can be reasonable for exploration, but it becomes expensive when the same logic must be changed in several places.

**Better direction:** Keep experiments separate from reusable project code.

- Put reusable data and model logic in `src/`.
- Put disposable experiments in `notebooks/` or `experiments/`.
- Use a small CLI for repeatable jobs.
- Promote only stable experiment code into the package.

For CLI work, [Typer](https://typer.tiangolo.com/) is a clean option. For serving APIs, [FastAPI](https://fastapi.tiangolo.com/) is a strong default. They solve different problems: Typer is for command-line workflows, while FastAPI is for HTTP services.

```text
.
|-- experiments/
|-- notebooks/
|-- src/
|   `-- project_name/
|       |-- data.py
|       |-- features.py
|       |-- train.py
|       `-- serve.py
`-- tests/
```

## 2. No High-Level Separation of Concerns

**Issue:** The ML package slowly collects responsibilities that do not belong together: data ingestion, model training, API serving, admin scripts, dashboards, deployment logic, and monitoring.

**Why it hurts:** When high-level responsibilities are tangled, a small change in one layer can break another. Cyclic dependencies appear between low-level utilities and high-level application code. Testing becomes difficult because everything imports everything else.

**Better direction:** Separate the system by responsibility.

- Data pipeline: ingestion, validation, transformation
- Model pipeline: training, evaluation, artifact creation
- Serving layer: inference API, batching, request validation
- Monitoring layer: metrics, logging, alerts, dashboards
- Orchestration layer: scheduled jobs, retries, deployment workflow

Docker, message queues, and microservices can help, but they are not the first solution. Start with clean module boundaries. Move to RabbitMQ, Kafka, Redis, Airflow, Kubernetes, or separate services only when the operational need is real.

## 3. No Low-Level Separation of Concerns

**Issue:** Inside a module, logic is still tangled. Preprocessing mutates global state. Training reads files directly. Inference depends on notebook-only objects. Business rules live inside model code.

**Why it hurts:** The code becomes hard to test because each function requires too much surrounding context. The model may work, but nobody can safely change the pipeline.

**Better direction:** Make individual units small and explicit.

- Use pure functions for deterministic transformations.
- Use classes for stateful components such as model wrappers, feature stores, clients, or caches.
- Keep I/O at the edges of the system.
- Pass dependencies explicitly instead of importing hidden global objects.
- Test transformations and post-processing without loading the full model when possible.

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class PredictionRequest:
    user_id: str
    text: str


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def build_features(request: PredictionRequest) -> dict[str, str]:
    return {
        "user_id": request.user_id,
        "normalized_text": normalize_text(request.text),
    }
```

Small pieces like this are not glamorous, but they are easy to test and easy to reuse.

## 4. No Configuration Data Model

**Issue:** Configuration is scattered across scripts, environment variables, notebooks, and hard-coded constants. Debugging becomes painful because nobody knows which values were used for a run.

**Better direction:** Treat configuration as a data model.

Pydantic is useful when configuration needs validation:

```python
from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    dataset_path: str
    model_name: str = "baseline"
    learning_rate: float = Field(gt=0)
    batch_size: int = Field(gt=0)
    max_epochs: int = Field(gt=0)
```

A good configuration object should answer:

- Which dataset was used?
- Which model, prompt, or feature version was used?
- Which hyperparameters were used?
- Which output artifact was produced?
- Which environment or deployment target was used?

Configuration is also part of reproducibility. If a result matters, the config should be saved with the artifact or experiment record.

## 5. Handling Legacy Models

**Issue:** Backward compatibility becomes painful when old models, old feature schemas, and old serving contracts are not versioned.

**Why it hurts:** A new model may require different features, labels, thresholds, or post-processing. If the system assumes one global schema, every old model becomes a special case.

**Better direction:** Version the model contract.

- Track model version, feature schema version, and output schema version.
- Keep adapters for legacy models instead of spreading compatibility logic everywhere.
- Use a model registry or artifact store for trained models.
- Preserve old evaluation results and config files.
- Validate rollback paths before a production release.

Scheduled jobs, dashboards, and terminal tools can help operations, but they are not the core solution. The core solution is a stable contract around each model artifact.

## 6. Code Quality Hygiene

**Issue:** Type hints, documentation, tests, complexity control, and dead-code removal are often treated as optional in ML projects.

**Why it hurts:** ML code already has uncertainty from data and models. The surrounding software should reduce uncertainty, not add more.

Useful tools include:

- `pytest` and `unittest` for tests
- `pytest-cov` or `coverage.py` for coverage
- `ruff`, `flake8`, or `pylint` for linting
- `mypy` for static type checking
- `pydantic` for runtime data validation
- `pydeps` for dependency visualization

Tooling is only useful if it supports a habit. A small but consistent baseline is better than a large tool stack nobody runs.

For a fuller project scaffold, see my Python project template post: [A Good Python Project Template to Use as a Starting Point](/writing/software/build-a-complete-python-project/). For model-specific checks, see [Testing in Machine Learning](/writing/blogs/mlops/ml-testing/).

## Summary

Quality ML code is not just clean Python. It is code that preserves the boundary between experimentation and production, keeps data and model contracts explicit, records configuration, supports legacy models, and makes failures easier to debug.

The practical test is simple: if the model changes tomorrow, can you update the system without rewriting the whole pipeline? If the answer is no, the codebase probably needs stronger boundaries before it needs another model.
