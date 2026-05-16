---
title: "A Good Python Project Template to Use as a Starting Point"
excerpt: "A practical Python project scaffold for packaging, testing, linting, documentation, and CI."
date: 2024/03/14
categories:
  - Software
tags:
  - Python Project
  - Optimization
layout: post
mathjax: true
toc: true
---

## Overview

When starting an open-source Python project, usefulness comes first. The project should solve a real problem. Right after that comes structure: how the codebase is organized, tested, documented, packaged, and maintained.

This post is a reference template for the parts I usually want in a serious Python project. Not every project needs every file. A small script should stay small. But it is easier to remove unnecessary pieces from a complete template than to bolt them on after the project has already grown.

The recommendations here are inspired by experienced open-source maintainers and Python packaging conventions. I am collecting the habits that make a project easier to install, review, test, and extend.

## Project Structure

A practical starting structure looks like this:

```text
.
├── .github/
│   └── workflows/
│       └── tests.yml
├── docs/
├── src/
│   └── your_package/
│       └── __init__.py
├── tests/
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── Makefile
├── README.md
└── pyproject.toml
```

The core pieces are:

- `src/`: package source code. The `src` layout helps catch import mistakes because tests must import the installed package, not a convenient local module path.
- `tests/`: unit, integration, and regression tests.
- `docs/`: user documentation, API references, examples, or design notes.
- `pyproject.toml`: project metadata, build configuration, dependencies, and tool configuration.
- `README.md`: the first user-facing explanation of what the project does and how to try it.
- `LICENSE`: the terms under which other people can use the code.
- `.github/workflows/`: CI checks for tests, linting, type checking, and packaging.
- `.pre-commit-config.yaml`: local checks that run before a commit.
- `Makefile`: short aliases for common commands.

Older projects often include `setup.py`, `setup.cfg`, `requirements.txt`, `MANIFEST.in`, and `.readthedocs.yml`. Those files can still be useful, but I would not add them by default. For a new package, start with `pyproject.toml` and add extra files only when a tool or workflow needs them.

## Environment

Start every project in a fresh virtual environment. It keeps dependencies isolated and makes debugging much less mysterious.

```shell
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

For a package managed through `pyproject.toml`, install the project in editable mode with development dependencies:

```shell
python -m pip install -e ".[dev]"
```

That assumes your `pyproject.toml` defines an optional `dev` dependency group:

```toml
[project]
name = "your-package"
version = "0.1.0"
description = "A short description of the project"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
  "pandas",
]

[project.optional-dependencies]
dev = [
  "pytest",
  "pytest-cov",
  "ruff",
  "mypy",
  "build",
  "twine",
]

[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"
```

Tools such as Poetry, Hatch, PDM, and uv can also manage environments and dependencies. The exact tool matters less than having one repeatable way to create the environment.

## Docker for Reproducible Development

For projects with heavier system dependencies, Docker can make the development environment more reproducible.

```dockerfile
ARG BASE_IMAGE=python:3.11-slim

FROM ${BASE_IMAGE} AS base

WORKDIR /opt/project
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --upgrade pip \
    && python -m pip install -e ".[dev]"
```

Build the image:

```shell
DOCKER_BUILDKIT=1 docker build -t your-package-dev -f Dockerfile .
```

Run a shell with the current project mounted:

```shell
docker run --rm -it -v "$(pwd)":/opt/project your-package-dev bash
```

Docker is not required for every project. It becomes useful when contributors need the same OS-level dependencies, CUDA libraries, database clients, or build tools.

## Testing and Coverage

A project should make correctness cheap to check. For most Python libraries, that starts with `pytest`.

Typical checks include:

- Unit tests for small functions and classes
- Integration tests for file I/O, API clients, database access, or model pipelines
- Regression tests for previously fixed bugs
- Coverage reports for important modules

Run tests with coverage:

```shell
pytest tests --cov=src --cov-report=term-missing
```

Coverage is useful, but it should not become theater. A high percentage does not guarantee useful tests. I care more about tests that cover important behavior, edge cases, and failure modes.

## Linting and Code Quality

Linters and formatters keep a project consistent. They also reduce review noise because contributors do not spend time debating whitespace, imports, or small style issues.

For new projects, I usually prefer `ruff` because it can replace several older tools in one fast command.

```shell
ruff check src tests
ruff format src tests
```

You can configure it in `pyproject.toml`:

```toml
[tool.ruff]
line-length = 88
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP"]
```

For type checking, `mypy` is still a common baseline:

```shell
mypy src
```

Type annotations are not enforced by the Python runtime, but they make code easier to read and easier to validate before release.

```python
from typing import Any


def sample_function(input_value: int) -> dict[str, Any]:
    """Convert an integer input into a structured result.

    Args:
        input_value: The integer to convert.

    Returns:
        A dictionary containing the original value and derived metadata.
    """
    return {"value": input_value, "is_positive": input_value > 0}
```

## Command Automation with Make

A `Makefile` gives contributors short, memorable commands for common tasks.

```makefile
.PHONY: install test lint format typecheck build clean

install:
	python -m pip install -e ".[dev]"

test:
	pytest tests --cov=src --cov-report=term-missing

lint:
	ruff check src tests

format:
	ruff format src tests

typecheck:
	mypy src

build:
	python -m build

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
```

Then the daily workflow becomes simple:

```shell
make lint
make typecheck
make test
```

This is not about `make` specifically. The important part is that the project has one obvious set of commands.

## Pre-Commit Checks

Pre-commit hooks catch small problems before they reach CI.

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-toml
      - id: debug-statements

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.11.2
    hooks:
      - id: mypy
```

Install the hooks:

```shell
pre-commit install
pre-commit run --all-files
```

Pre-commit checks should be fast. If a check takes too long, put it in CI or make it an explicit release command.

## GitHub Actions

CI should run the checks that reviewers care about before they review the code.

```yaml
name: Tests

on:
  push:
  pull_request:

jobs:
  tests:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install package
        run: make install

      - name: Lint
        run: make lint

      - name: Type check
        run: make typecheck

      - name: Test
        run: make test
```

For a library, I would eventually add a matrix across supported Python versions. For a small project, one supported version is enough until the maintenance burden is worth it.

## Summary

A good Python project template should make the common path obvious:

- Install the project
- Run tests
- Format and lint code
- Type-check important modules
- Build the package
- Publish documentation
- Run the same checks in CI

The point is not to add ceremony. The point is to make the project easy for future you, and for other contributors, to trust.

## References

- [Python Packaging User Guide: Writing your pyproject.toml](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)
- [Eugene Yan: Setting up a Python project for automation and collaboration](https://eugeneyan.com/writing/setting-up-python-project-for-automation-and-collaboration/#set-up-a-virtualenv-and-install-packages)
- [Tian Gao](https://github.com/gaogaotiantian)
