---
title: "A good Python projecttemplate to use as starting point"
excerpt: "A summary of many masterminds' projects style"
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

When you build up an open source Python project, the first thing to keep in mind is its usefulness. It should really solve some unsolved problems. And right after this, is how you plan and build up a solution step by step. Designing the codebase is one big part of the planning. Everyone open source contributer should have his or her own way of starting a project, but not everyone is good at maintaining the codebase in the long run. One big reason for that is they lack the mindset of building a robust architecture for their code. In this blog, I will cover the major parts every python project (at least for me) should include. These components can be redundant. But redundancy is good to me, as subtraction is easier than addition especially when it comes to simple things.

Heads up that this is more for references. Most of the content are inspired from other coders on GitHub who have accumulated the good habit of building robust codebase template over the years. I\'m here just to learn, ABSORB and implement them in my own style.

## Project structure

The first thing is actually codebase structure. My template structure looks like the following:

```
├── .github
│   └── workflows
├── .pre-commit-config.yaml
├── LICENSE
├── README.md
├── .gitignore
├── pyproject.toml
├── .readthedocs.yml
├── MANIFEST.in
├── Makefile
├── requirements.txt
├── setup.py
├── docs
├── tests
└── src
```

Here:

- `src` contains your project code,
- `tests` include all your test cases,
- `docs` include your documentation of the code (very often used as the api doc, a common choice is `readthedoc` with `sphinx`).
- `.gitignore, README.md, pyproject.toml, .readthedocs.yml, MANIFEST.in` are config and instruction files for you to setup your project and potentially make it into pypl. This also enables quick prototying of environment which I\'ll talk about later.
- `requirements.txt` keeps a list of major packages/dependencies. This can be split into prod/dev/etc as you want.
- `Makefile` is to automate many lengthy command-line code into short `make xxx` command.
- `.github/workflow` is the CI/CD pipeline based on GitHub Actions/Jenkins and other integration tools
- `.pre-commit-config.yaml` is the file the carries out pre-commit code validations
- `LICENSE` is just there in case your code will be used by many people (which is gooood!)

## Environment

A basic thing when it comes to setup environment is to always start with a new virtual environment. This helps separate dependencies for different projects. There are three major ways.

1. direcly using virtualenv

```shell
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

2. Use `poetry` dependency manager

```python
# Create a poetry project
poetry init --no-interaction

# Add numpy as dependency
poetry add numpy

# Recreate the project based on the pyproject.toml
poetry install

# To get the path to poetry venv (for PyCharm)
poetry env info
```

With our (virtual) environment set up and activated, we can proceed to install python packages. To keep our production environment as light as possible, we’ll want to separate the packages needed for dev and prod:

dev: These are only used for development (e.g., testing, linting, etc.) and are not required for production.
prod: These are needed in production (e.g., data processing, machine learning, etc.).

```shell
# Install dev packages which we'll use for testing, linting, type-checking etc.
pip install pytest pytest-cov pylint mypy codecov

# Freeze dev requirements
pip freeze > requirements.dev

# Install prod packages
pip install pandas

# Freeze dev requirements
pip freeze > requirements.prod
```

3. use Docker as a Dev Environment instead

```dockerfile
ARG BASE_IMAGE=python:3.8

FROM ${BASE_IMAGE} as base

LABEL maintainer='eugeneyan <dev@eugeneyan.com>'

# Use the opt directory as our dev directory
WORKDIR /opt

ENV PYTHONUNBUFFERED TRUE

COPY requirements.dev .
COPY requirements.prod .

# Install python dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir wheel \
    && pip install --no-cache-dir -r requirements.dev \
    && pip install --no-cache-dir -r requirements.prod \
    && pip list
```

then run

```shell
DOCKER_BUILDKIT=1 docker build -t dev -f Dockerfile .
```

to setup the environment.
Now we can run `bash` using the docker container (after mounting current dir to `/opt` folder) via

```shell
docker run --rm -it -v $(pwd):/opt bash
```

## testing + coverage

You\'ll need to thoroughly test your code. In generally, when your codebase involve little infra dependency, and isn\'t heavy on data/model ingestion/output, most of tests will still be unit tests. `pytest` and `unittests` are your best friends. In the meantime, never forget to ensure high test coverage with the help of `coveragepy` and `codecov`. A little badge can be attached to `README.md` (check this [guide](https://stackoverflow.com/questions/54010651/how-to-display-codecov-io-badge-in-github-readme-md)) once you\'ve robustly tested your code, which actually suggests that you\'re a responsible engineer!

## linting + code quality

1. ensuring code consistency with linting
   Linters analyze code to flag proramming errors, bugs, and deviations from standards. Linting leads to good code quality. As you use linting to correctly format your code, you are forming the good having of following a good coding style as well.

   Many people use either `pylint` and `flake8` for linting. Note that very often you may want to ignore certain patterns in linting standard that your project doesn\'t agree with. There are many ways to configure it properly. In the current folder structure I suggested, you should edit in `pyproject.toml`. Note, for `flake8`, you need to install `flake8-pyproject` package independently for the configuration to work property.

2. python coding style (PEP8)
   Sometimes linting also helps to check with particular function format. For example, `pylint` and `flake8` both require function/class annotations. But you\'ll need to style your annotations correctly:

```python
def sample_function(input1: int) -> Any:
    """Description of what the function does

    Args:
        input1: some input explanation

    Returns:
        Some output explanation
    """
    ...
```

3. typing checks
   The Python runtime does not enforce type annotations; it’s dynamically typed and only verifies types (at runtime) via duck typing. Nonetheless, we can use type annotations and a static type checker to verify the type correctness of our code. `mypy` is the most widely adopted option from my impression. In the meantime, consider using `pydantic` for a more OOP styled typing.

## command automation with make

`Makefile` is a life-saver when we have many tasks to run before committing code. When each task is a long command-line code, it can be hard to remember and go through your terminal history to retrieve it. Hence we can just define tasks using the `Makefile` as follows:

```shell
.PHONY: refresh build install build_dist json release lint test clean

refresh: clean build install lint

build:
	python -m build

install:
	pip install .

build_dist:
	make clean
	python -m build
	pip install dist/*.whl
	make test

json:
	python example/generate_examples.py

release:
	python -m twine upload dist/*

lint:
	flake8 src/ tests/ example/ --exclude "src/db/*" --count --statistics
	mypy src/ --exclude 'src/.*'

test:
	. .venv/bin/activate && py.test tests --cov=src --cov-report=term-missing --cov-fail-under 95

clean-pyc:
    find . -name '*.pyc' -exec rm -f {} +
    find . -name '*.pyo' -exec rm -f {} +
    find . -name '*~' -exec rm -f {} +
    find . -name '__pycache__' -exec rm -fr {} +

clean-test:
    rm -f .coverage
    rm -f .coverage.*

clean: clean-pyc clean-test
```

and run any of `make test`, `make clean`, `make lint` tasks to complete the command-line code defined under it.

## pre-commit and pre-push

When it comes down to committing your code and pushing your code (via PR for e.g.), things get complicated as people often make mistakes in linting/typing/formatting. When it happends, pre-commit automations and checks help us resolve those problems to some extent. We can use a `.pre-commit-config.yaml` file to help us make corrections and find errors asap (may have overlap with the make-automation method) before we even commit our code to local branch.

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: debug-statements
      - id: double-quote-string-fixer
      - id: name-tests-test
      - id: requirements-txt-fixer
  - repo: https://github.com/asottile/setup-cfg-fmt
    rev: v2.5.0
    hooks:
      - id: setup-cfg-fmt
  - repo: https://github.com/asottile/reorder-python-imports
    rev: v3.12.0
    hooks:
      - id: reorder-python-imports
        exclude: ^(pre_commit/resources/|testing/resources/python3_hooks_repo/)
        args: [--py39-plus, --add-import, "from __future__ import annotations"]
  - repo: https://github.com/asottile/add-trailing-comma
    rev: v3.1.0
    hooks:
      - id: add-trailing-comma
  - repo: https://github.com/asottile/pyupgrade
    rev: v3.15.1
    hooks:
      - id: pyupgrade
        args: [--py39-plus]
  - repo: https://github.com/hhatto/autopep8
    rev: v2.0.4
    hooks:
      - id: autopep8
  - repo: https://github.com/PyCQA/flake8
    rev: 7.0.0
    hooks:
      - id: flake8
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.9.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
        exclude: ^testing/resources/
```

For push actions, we can define a check in the `.github/workflow/test.yml` to run the checks:

```yaml
# .github/workflows/tests.yml
name: Tests
on: push
jobs:
  tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v1
        with:
          python-version: 3.8
          architecture: x64
      - run: make setup-venv
      - run: make checks
```

This helps us find problems before reviewing any code to start with.

## Summary

These are the first few steps I take to build up a robust codebase. It came a long way from learning the codebase and blogs from smart and selfless engineers who\'re willing to share their experience and code online. I\'m truly grateful of these people.

## References

1. [Eugene Yan](https://eugeneyan.com/writing/setting-up-python-project-for-automation-and-collaboration/#set-up-a-virtualenv-and-install-packages)
2. [Tian Gao](https://github.com/gaogaotiantian)
