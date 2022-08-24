---
title: "Regression Models: Logistic Regression"
excerpt: "Logit Model: the simple becomes the powerful"
layout: post
date: 2019/05/13
updated: 2021/4/20
categories:
  - Blogs
tags: 
  - Regression
  - Supervised Learning
mathjax: true
toc: true
---
### Definition

- We have a mathematical function which gives a value between $-\infty$ and $\infty$, and to convert it to a value between (0,1), we need a <b>Sigmoid</b> function or a logistic function
- We can visualize it as a boundary (the decision boundary) to separate 2 categories on a hyperplane, where each dimension is a variable (a certain type of information)
- The algorithm used is also *gradient descent*

### Common Questions
1. What is a logistic function?   
    __Answer__: $f(z) = {1\over (1+e -z) }$.  
2. What is the range of values of a logistic function?  
    __Answer__: The values of a logistic function will range from 0 to 1. The values of Z will vary from $-\infty$ to $\infty$.  
3. What are the cost functions of logistic function?    
    __Answer__: The popular 2 are __Cross-entropy__ or __log loss__. Note that __MSE__ is not used as squaring sigmoid violates convexity (cause local extrema to appear).

### Basic Implementation
```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
X, y = load_iris(return_X_y=True)
clf = LogisticRegression(random_state=2).fit(X, y)
clf.predict(X[:2, :])

clf.predict_proba(X[:2, :])
clf.score(X, y)
```

### Notes
In fact, logistic regression is simple, but the key thing here is actually on the mathematics behind *gradient descent* and its multi-dimensional variations. I\'ll discuss about them in future posts.