---
title: "Some Tricks in Real-World Machine Learning Engineering"
date: 2024-03-02
categories:
  - Blogs
  - Draft Notes
tags:
  - MLOps
  - Feature Engineering
excerpt: "Practical notes on moving from notebooks to pipelines, handling missing values, scaling features, encoding categories, and keeping ML code production-friendly."
layout: post
---

## Introduction

Real-world machine learning engineering is often less about clever algorithms and more about small habits that prevent messy systems.

This post collects practical tricks I keep reaching for:

- Move experiments out of notebooks before they become production code.
- Treat feature engineering as a versioned pipeline.
- Handle missing values deliberately.
- Scale and encode features with training-serving consistency.
- Keep model code easy to test and rerun.

## Convert Notebooks Into Scripts Early

Notebooks are excellent for exploration. They are poor as the long-term source of truth for a training pipeline.

A useful pattern is:

1. Explore in a notebook.
2. Move reusable logic into Python modules.
3. Keep the notebook as a report or scratchpad.
4. Run training through a script or CLI.

For a quick conversion:

```bash
jupyter nbconvert --to script train_model.ipynb
```

Then clean the generated script into functions:

```python
def load_data(config):
    ...


def build_features(data, config):
    ...


def train_model(features, labels, config):
    ...


def main(config):
    data = load_data(config)
    features, labels = build_features(data, config)
    model = train_model(features, labels, config)
    return model
```

The goal is not to ban notebooks. The goal is to keep production behavior in code that can be tested, reviewed, and rerun.

## Handle Missing Values by Cause

Missing values are not all the same.

Ask why the value is missing:

- **Missing completely at random:** the absence is unrelated to the value or target.
- **Missing because of another variable:** for example, a field appears only for a certain product or user type.
- **Missing because of the value itself:** for example, users with a sensitive condition may skip a field.
- **Missing because of pipeline failure:** the feature should exist, but extraction failed.

Different causes need different treatment:

- Drop rows only when the lost data is small and unbiased.
- Drop columns when the feature is mostly unavailable or unreliable.
- Impute with mean, median, mode, or learned values when appropriate.
- Add missingness indicators when missingness itself carries signal.
- Fail the pipeline when missingness indicates a broken upstream dependency.

The last case is important. Not every missing value should be "handled"; some should page the owner.

## Scale Features Consistently

Scaling should be fit on training data and reused everywhere else.

Bad pattern:

```python
train["x"] = (train["x"] - train["x"].mean()) / train["x"].std()
test["x"] = (test["x"] - test["x"].mean()) / test["x"].std()
```

Better pattern:

```python
from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()
train_x = scaler.fit_transform(train[feature_cols])
valid_x = scaler.transform(valid[feature_cols])
test_x = scaler.transform(test[feature_cols])
```

The same fitted scaler must be available in inference. Otherwise training and serving will disagree.

## Encode Categories With Future Values in Mind

Categorical features cause production issues because new categories appear after deployment.

Options:

- **One-hot encoding:** simple, but needs a strategy for unknown categories.
- **Ordinal encoding:** compact, but can introduce fake ordering.
- **Target encoding:** powerful, but easy to leak labels if done incorrectly.
- **Hashing:** maps categories into a fixed number of buckets and naturally handles unseen categories.
- **Embeddings:** useful for high-cardinality categorical features in deep models.

Hashing is a strong industrial trick:

```python
from sklearn.feature_extraction import FeatureHasher


hasher = FeatureHasher(n_features=2**18, input_type="string")
features = hasher.transform(user_category_lists)
```

Collisions happen, but a fixed-size hashed representation avoids the "unknown category broke inference" problem.

## Keep Feature Pipelines Versioned

Feature logic should be treated like model code.

Track:

- Feature definitions.
- Training data version.
- Transformation code version.
- Fitted preprocessing artifacts.
- Schema.
- Expected value ranges.

When a model changes behavior, feature drift is often the culprit. Versioning makes the investigation possible.

## Do Not Hide Data Leakage

Leakage often looks like great performance.

Watch for:

- Features created after the prediction time.
- Aggregates that include the target period.
- Duplicate users or events across train and test.
- Target-derived text or labels.
- Preprocessing fit on the full dataset before splitting.

Use time-based splits when the production task is time-based. Random splits can be misleading when future information leaks into training.

## Closing

Small ML engineering habits compound. Move code out of notebooks, version preprocessing, treat missing values by cause, make feature transforms reusable, and assume production data will surprise you.

These tricks are not glamorous, but they are the difference between a model that works once and a system that keeps working.
