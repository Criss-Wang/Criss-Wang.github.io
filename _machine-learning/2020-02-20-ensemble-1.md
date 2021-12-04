---
title: "Ensemble Models: Bagging Techniques"
date: 2021-01-14
layout: single
author_profile: true
categories:
  - Supervised Learning
tags: 
  - Bagging
  - Random Forest
excerpt: "Gather all we have!"
mathjax: "true"
---
## Overview
We have learnt about what bagging is in *Ensemble Models: Overview*, to recap, bagging is:
- In bagging (Bootstrap Aggregating), a set of weak learners are combined to create a strong learner that obtains better performance than a single one.
- Bagging helps to decrease the model’s variance.
- Combinations of multiple classifiers decrease variance, especially in the case of unstable classifiers, and may produce a more reliable classification than a single classifier.
In this blog, we will use random forest as an example to illustrate how bagging works
**Bagging** works as follows:-
1. Multiple subsets are created from the original dataset, selecting observations with replacement.
2. A base model (weak model) is created on each of these subsets.
3. The models run in parallel and are independent of each other.
4. The final predictions are determined by combining the predictions from all the models.

Next let's consider random forest, a model that fully utilized the idea of bagging in its procedure.

## Random Forest
### 1. Definition
- A random forest consists of multiple random decision trees. Two types of randomnesses are built into the trees. 
    - First, each tree is built on a random sample from the original data. 
    - Second, at each tree node, a subset of features are randomly selected to generate the best split. (Key difference from Bagging algorithms)
- An ensemble model that is widely applied (as it can be parallelized)
- Designed to solve the overfitting issue in Decision Tree. The idea is that by training each tree on different samples, although each tree might have high variance with respect to a particular set of the training data, overall, the entire forest will have lower variance but not at the cost of increasing the bias.
- __Procedure__  
    Execute until every last combination is exhausted 
    1. Bootstrapping: Create a bootstraped dataset: randomly select samples from the dataset until it reaches the same size as the original sample (we're allowed to pick the same sample more than once);
    2. Decision Tree Construction: Create a decision tree using the bootstraped dataset, but only use a random subset of variables/features at each step (i.e each decision node selection);
    3. **Bagging**: defined as bootstrapping the data plus using the aggregation to make a decision
        - given a new instance, run through all the decision trees (the enitre random forest) and obtain the sum of votes for y = 1 and y = 0; (this step is called "aggregation")
        - decide on result of aggregation => using the result with higher vote;
- Choose the most accurate random forest:
    1. Measure Accuracy based on the Out-of-Bag samples (CV) => compute the Out-of-Bag Error as the # of samples which Bagging classifies wrongly
    2. Choice of number of variable used per step affects accuracy (optimized during CV): Usually choose the square root of the # of total variables and try a few settings above and below that value 
- The low correlation between models is the key
> __WARNING__: RF is often considered as __Bagging__ model while it is __not always true__, [see this link](https://stats.stackexchange.com/questions/264129/what-is-the-difference-between-bagging-and-random-forest-if-only-one-explanatory)

### 2. Pros & Cons
**Pros**
- The power of handle large data sets with higher dimensionality (as each tree select much less features in its construction)
- The model outputs importance of variable, which can be a very handy feature (`rf.feature_importance_`)
- Balancing errors in data sets where classes are imbalanced.
- It has an effective method for estimating missing data and maintains accuracy when large proportion of the data are missing.
- Using the out of bag error estimate for selection the most accurate random forest removes the need for a set aside test set.

**Cons**
- It has very poor interpretability
- Does not work well for extrapolation to predict for data that is outside of the bounds of your original training data
- Random forest can feel like a black box approach for a statistical modelers we have very little control on what the model does. You can at best try different parameters and random seeds.

### 3. Simple Implementation
This is a template inspired by the Kaggle notebooks. I shall thank those writers whose code I borrowed from. 

Also note that here an aws s3 connection is made, which automatically makes the process parallelized.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

RSEED = 50

# Load in data
df = pd.read_csv('https://s3.amazonaws.com/projects-rf/clean_data.csv')

# Full dataset: https://www.kaggle.com/cdc/behavioral-risk-factor-surveillance-system

# Extract the labels
labels = np.array(df.pop('label'))

# 30% examples in test data
train, test, train_labels, test_labels = train_test_split(df,
                                         labels, 
                                         stratify = labels,
                                         test_size = 0.3, 
                                         random_state = RSEED)

# Imputation of missing values
train = train.fillna(train.mean())
test = test.fillna(test.mean())

# Features for feature importances
features = list(train.columns)

# Create the model with 100 trees
model = RandomForestClassifier(n_estimators=100, 
                               random_state=RSEED, 
                               max_features = 'sqrt',
                               n_jobs=-1, verbose = 1)

# Fit on training data
model.fit(train, train_labels)


n_nodes = []
max_depths = []

# Stats about the trees in random forest
for ind_tree in model.estimators_:
    n_nodes.append(ind_tree.tree_.node_count)
    max_depths.append(ind_tree.tree_.max_depth)
    
print(f'Average number of nodes {int(np.mean(n_nodes))}')
print(f'Average maximum depth {int(np.mean(max_depths))}')

# Training predictions (to demonstrate overfitting)
train_rf_predictions = model.predict(train)
train_rf_probs = model.predict_proba(train)[:, 1]

# Testing predictions (to determine performance)
rf_predictions = model.predict(test)
rf_probs = model.predict_proba(test)[:, 1]

from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Plot formatting
plt.style.use('fivethirtyeight')
plt.rcParams['font.size'] = 18

def evaluate_model(predictions, probs, train_predictions, train_probs):
    """Compare machine learning model to baseline performance.
    Computes statistics and shows ROC curve."""
    
    baseline = {}
    
    baseline['recall'] = recall_score(test_labels, 
                                     [1 for _ in range(len(test_labels))])
    baseline['precision'] = precision_score(test_labels, 
                                      [1 for _ in range(len(test_labels))])
    baseline['roc'] = 0.5
    
    results = {}
    
    results['recall'] = recall_score(test_labels, predictions)
    results['precision'] = precision_score(test_labels, predictions)
    results['roc'] = roc_auc_score(test_labels, probs)
    
    train_results = {}
    train_results['recall'] = recall_score(train_labels, train_predictions)
    train_results['precision'] = precision_score(train_labels, train_predictions)
    train_results['roc'] = roc_auc_score(train_labels, train_probs)
    
    for metric in ['recall', 'precision', 'roc']:
        print(f'{metric.capitalize()} Baseline: {round(baseline[metric], 2)} Test: {round(results[metric], 2)} Train: {round(train_results[metric], 2)}')
    
    # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(test_labels, [1 for _ in range(len(test_labels))])
    model_fpr, model_tpr, _ = roc_curve(test_labels, probs)

    plt.figure(figsize = (8, 6))
    plt.rcParams['font.size'] = 16
    
    # Plot both curves
    plt.plot(base_fpr, base_tpr, 'b', label = 'baseline')
    plt.plot(model_fpr, model_tpr, 'r', label = 'model')
    plt.legend();
    plt.xlabel('False Positive Rate'); 
    plt.ylabel('True Positive Rate'); plt.title('ROC Curves');
    plt.show();

evaluate_model(rf_predictions, rf_probs, train_rf_predictions, train_rf_probs)

```