---
title: "Clustering: K Means and Gaussian Mixture Models"
date: 2020-02-01
layout: single
author_profile: true
categories:
  - Unsupervised Learning
tags: 
  - Clustering
  - Expectation Maximization
excerpt: "The basic models for clustering"
mathjax: "true"
---
## Overview
In this blog we talk about K means and GMM algorithms, the famous and intuitively useful algorithms. As we venture further into unsupervised learning/clustering problems, we will see more interesting problem formulations as well as diverse evaluation metrics. Hope we would enjoy this learning journey along the way :)
## K means
### 1. Defintion
- `Clustering`: A cluster refers to a collection of data points aggregated together because of certain similarities. 
- `K-means`: an iterative algorithm that tries to partition the dataset into  𝐾  pre-defined distinct non-overlapping subgroups (clusters) where each data point belongs to **only one group**
- Kmeans gives more weight to the bigger clusters.
- Kmeans assumes spherical shapes of clusters (with radius equal to the distance between the centroid and the furthest data point) and doesn't work well when clusters are in different shapes such as elliptical clusters.
- __Full procedure__: 
    1. Specify number of clusters $K$.
    2. Initialize centroids by first shuffling the dataset and then randomly selecting $K$ data points for the centroids without replacement.
    3. Keep iterating until there is no change to the centroids. i.e assignment of data points to clusters isn't changing.
        - Compute the sum of the squared distance between data points and all centroids.
        - Assign each data point to the closest cluster (centroid).
        - Compute the centroids for the clusters by taking the average of the all data points that belong to each cluster.
- `Expectation-Maximization`: The approach kmeans follows to solve the problem is called **Expectation-Maximization**. The EM algorithm attempts to find maximum likelihood estimates for models with latent variables.   The E-step is assigning the data points to the closest cluster. The M-step is computing the centroid of each cluster. Below is a break down of how we can solve it mathematically (feel free to skip it).
    - The objective function is: $J = \sum_{i = 1}^{m}\sum_{k = 1}^{K}w_{ik}\|x^i - \mu_k\|^2$  
    where $w_{ik} = 1$ for data point $x^i$ if it belongs to cluster $k$; otherwise, $w_{ik} = 0$. Also, $\mu_k$ is the centroid of $x^i$'s cluster.

    - It's a minimization problem of two parts. We first minimize J w.r.t. $w_{ik}$ and treat $\mu_k$ fixed. Then we minimize J w.r.t. $\mu_k$ and treat $w_{ik}$ fixed. Technically speaking, we differentiate J w.r.t. $w_{ik}$ first and update cluster assignments  (*E-step*). Then we differentiate J w.r.t. $\mu_{k}$ and recompute the centroids after the cluster assignments from previous step  (*M-step*). Therefore, E-step is:
  
        $\frac{\partial J}{\partial w_{ik}} = \sum_{i = 1}^{m}\sum_{k = 1}^{K}\|x^i - \mu_k\|^2$
        {: style="text-align: center;"}
        $$ \begin{equation}
          \Rightarrow w_{ik} = \begin{cases}
            1 & \text{if $k = arg min_j\ \|x^i - \mu_j\|^2$}\\
            0 & \text{otherwise}.
          \end{cases}
        \end{equation}
        $$
    In other words, assign the data point $x^i$ to the closest cluster judged by its sum of squared distance from cluster's centroid.

    And M-step is:
    
    $\frac{\partial J}{\partial \mu_k} = 2\sum_{i = 1}^{m}w_{ik}(x^i - \mu_k) = 0$
    $\Rightarrow \mu_k = \frac{\sum_{i = 1}^{m}w_{ik}x^i}{\sum_{i = 1}^{m}w_{ik}}\tag{2}$
    {: style="text-align: center;"}
    Which translates to recomputing the centroid of each cluster to reflect the new assignments.
- Standardization:
    - Since clustering algorithms including kmeans use distance-based measurements to determine the similarity between data points, it's recommended to standardize the data to have a mean of zero and a standard deviation of one since almost always the features in any dataset would have different units of measurements such as age vs income.
- Cold start the code may lead to __Local optimum__: Need to use different initializations of centroids and pick the results of the run that that yielded the lower sum of squared distance.
- __Evaluation Method__:
    Contrary to supervised learning where we have the ground truth to evaluate the model's performance, clustering analysis doesn't have a solid evaluation metric that we can use to evaluate the outcome of different clustering algorithms. Moreover, since kmeans requires $k$ as an input and doesn't learn it from data, there is no right answer in terms of the number of clusters that we should have in any problem. Sometimes domain knowledge and intuition may help but usually that is not the case. In the cluster-predict methodology, we can evaluate how well the models are performing based on different $K$ clusters since clusters are used in the downstream modeling.

    In this notebook we'll cover two metrics that may give us some intuition about $k$:
    - Elbow method
     
      **Elbow method** gives us an idea on what a good $k$ number of clusters would be based on the sum of squared distance (SSE) between data points and their assigned clusters' centroids. We pick $k$ at the spot where SSE starts to flatten out and forming an elbow. We'll use the geyser dataset and evaluate SSE for different values of $k$ and see where the curve might form an elbow and flatten out.
    - Silhouette analysis
     
      **Silhouette analysis** can be used to determine the degree of separation between clusters. For each sample:
      - Compute the average distance from all data points in the same cluster ($a^i$).
      - Compute the average distance from all data points in the closest cluster ($b^i$).
      - Compute the coefficient:
      $\frac{b^i - a^i}{max(a^i, b^i)}$
      The coefficient can take values in the interval [-1, 1].
      - If it is 0 --> the sample is very close to the neighboring clusters.
      - It it is 1 --> the sample is far away from the neighboring clusters.
      - It it is -1 --> the sample is assigned to the wrong clusters.

      Therefore, we want the coefficients to be as big as possible and close to 1 to have a good clusters. We'll use here geyser dataset again because its cheaper to run the silhouette analysis and it is actually obvious that there is most likely only two groups of data points.

### 2. Pros & Cons
**Pros**
1. Easy to interpret
2. Relatively fast
3. Scalable for large data sets
4. Able to choose the positions of initial centroids in a smart way that speeds up the convergence
5. Guarantees convergence

**Cons**
1. The globally optimal result may not be achieved
2. The number of clusters must be selected beforehand
3. k-means is limited to linear cluster boundaries:
    - this one may be solved using Similar technique as SVM does.
    - One possible solution is the "Spectral Clustering": i.e Kernelized K-means below

### 3. Applications
- Not to use if it contains heavily overlapping data/full of outliers
- Not so well if there are many categorical fields  
- Not so well if the clusters have a complicated geometric shapes  

__Real-world samples__
- Market Segmentation
- Document clustering
- Image segmentation
- Image compression 

### 4. Code Implementation
```python
from sklearn.cluster import KMeans

km = KMeans(n_clusters=2, max_iter=100)
km.fit(X_std)
centroids = km.cluster_centers_
```

## GMM
### 1. Definition
- The Gaussian mixture model (GMM) can be regarded as an optimization of the k-means model. It is not only a commonly used in industry but also a generative model.
- A model composed of K single Gaussian models. These K submodels are the hidden variables of the hybrid model
    - `Single Gaussian model`: A __Univariate Gaussian Distribution__ for the data 
- Attempts to find a mixed representation of the probability distribution of the multidimensional Gaussian model, thereby fitting a data distribution of arbitrary shape.
- Also uses __EM__ algorithm<sup>[^1]</sup>:
  
  [^1]:https://stephens999.github.io/fiveMinuteStats/intro_to_em.html
  - if our observations $X_{i}$ come from a mixture model with $K$ mixture components, the marginal probability distribution of $X_{i}$ is of the form:

    $$
    P\left(X_{i}=x\right)=\sum_{k=1}^{K} \pi_{k} P\left(X_{i}=x \mid Z_{i}=k\right)
    $$

    where $Z_{i} \in\{1, \ldots, K\}$ is the latent variable representing the mixture component for $X_{i}, P\left(X_{i} \mid Z_{i}\right)$ is the mixture component, and $\pi_{k}$ is the mixture proportion representing the probability that $X_{i}$ belongs to the $k$-th mixture component.

    Let $N\left(\mu, \sigma^{2}\right)$ denote the probability distribution function for a normal random variable. In this scenario, we have that the conditional distribution $X_{i} \mid Z_{i}=k \sim N\left(\mu_{k}, \sigma_{k}^{2}\right)$ so that the marginal distribution of $X_{i}$ is: 

    $$
    P\left(X_{i}=x\right)=\sum_{k=1}^{K} P\left(Z_{i}=k\right) P\left(X_{i}=x \mid Z_{i}=k\right)=\sum_{k=1}^{K} \pi_{k} N\left(x ; \mu_{k}, \sigma_{k}^{2}\right)
    $$

    Similarly, the joint probability of observations $X_{1}, \ldots, X_{n}$ is therefore:

    $$
    P\left(X_{1}=x_{1}, \ldots, X_{n}=x_{n}\right)=\prod_{i=1}^{n} \sum_{k=1}^{K} \pi_{k} N\left(x_{i} ; \mu_{k}, \sigma_{k}^{2}\right)
    $$

    This note describes the EM algorithm which aims to obtain the maximum likelihood estimates of $\pi_{k}, \mu_{k}$ and $\sigma_{k}^{2}$ given a data set of observations $\{x_{1}, \ldots x_{n}\}$.
  - Likelihood expression is $P(X, Z \mid \mu, \sigma, \pi)=\prod_{i=1}^{n} \prod_{k=1}^{K} \pi_{k}^{I\left(Z_{i}=k\right)} N\left(x_{i} \mid \mu_{k}, \sigma_{k}\right)^{I\left(Z_{i}=k\right)}$; Take log, compute and simplify the expected value of the complete log-likelihood:

    $$
    \begin{aligned}
    E_{Z \mid X}[\log (P(X, Z \mid \mu, \sigma, \pi))] &=E_{Z \mid X}\left[\sum_{i=1}^{n} \sum_{k=1}^{K} I\left(Z_{i}=k\right)\left(\log \left(\pi_{k}\right)+\log \left(N\left(x_{i} \mid \mu_{k}, \sigma_{k}\right)\right)\right)\right] \\
    &=\sum_{i=1}^{n} \sum_{k=1}^{K} E_{Z \mid X}\left[I\left(Z_{i}=k\right)\right]\left(\log \left(\pi_{k}\right)+\log \left(N\left(x_{i} \mid \mu_{k}, \sigma_{k}\right)\right)\right) \\
    &=\sum_{i=1}^{n} \sum_{k=1}^{K} \gamma_{Z_{i}}(k)\left(\log \left(\pi_{k}\right)+\log \left(N\left(x_{i} \mid \mu_{k}, \sigma_{k}\right)\right)\right)
    \end{aligned}
    $$
  - From here, we derive the expressions for each parameter:
    
    $$\sum_{i=1}^{n} \gamma_{Z_{i}}(k) \frac{\left(x_{i}-\mu_{k}\right)}{\sigma_{k}^{2}}=0 \tag{2}$$

    $$
    \hat{\mu_{k}}=\frac{\sum_{i=1}^{n} \gamma_{z_{i}}(k) x_{i}}{\sum_{i=1}^{n} \gamma_{z_{i}}(k)}=\frac{1}{N_{k}} \sum_{i=1}^{n} \gamma_{z_{i}}(k) x_{i}\tag{3}
    $$

    $$
    \begin{align}
    \hat{\sigma_{k}^{2}} &=\frac{1}{N_{k}} \sum_{i=1}^{n} \gamma_{z_{i}}(k)\left(x_{i}-\mu_{k}\right)^{2}\tag{4}\\
    \hat{\pi_{k}} &=\frac{N_{k}}{n}\tag{5}
    \end{align}
    $$
  - The EM algorithm, motivated by the two observations above, proceeds as follows:
    1. Initialize the $\mu_{k}$ 's, $\sigma_{k}$ 's and $\pi_{k}$ 's and evaluate the log-likelihood with these parameters.
    2. E-step: Evaluate the posterior probabilities $\gamma_{Z_{i}}(k)$ using the current values of the $\mu_{k}$ 's and $\sigma_{k}$ 's with equation (2)
    3. M-step: Estimate new parameters $\hat{\mu_{k}}, \hat{\sigma_{k}^{2}}$ and $\hat{\pi_{k}}$ with the current values of $\gamma_{Z_{i}}(k)$ using equations (3), (4) and (5).
    4. Evaluate the log-likelihood with the new parameter estimates. If the loglikelihood has changed by less than some small $\epsilon$, stop. Otherwise, go back to step 2 .

    The EM algorithm is sensitive to the initial values of the parameters, so care must be taken in the first step. However, assuming the initial values are "valid," one property of the EM algorithm is that the log-likelihood increases at every step. This invariant proves to be useful when debugging the algorithm in practice.

### 2. Pros & Cons
**Pros**
- GMM is a lot more flexible in terms of cluster covariance
- It is a soft-clustering method, which assign sample membersips to multiple clusters. This characteristic makes it the fastest algorithm to learn mixture models

**Cons**
- Slower than k-means
- does not work if the mixture is not really a gaussian distribution
- It is very sensitive to the initial values which will condition greatly its performance.
- GMM may converge to a local minimum, which would be a sub-optimal solution.
- When having insufficient points per mixture, the algorithm diverges and finds solutions with infinite likelihood unless we regularize the covariances between the data points artificially.

### 3. Application
- perform GMM when you know that the data points are mixtures of a gaussian distribution
- if you think that your model is having some hidden, not observable parameters, then you can try to use GMM. 

### 4. Simple code
```python
from sklearn.mixture import GaussianMixture 
gmm = GaussianMixture(n_components = 3) 
gmm.fit(X_principal)
gmm.fit_predict(X_principal)
```