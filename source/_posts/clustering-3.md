---
title: "Clustering: DBSCAN"
layout: post
date: 2020/02/06
updated: 2021/9/1
categories:
  - Blogs
tags: 
  - Machine Learning
  - Clustering
  - Unsupervised Learning
excerpt: "Graphical methods continued..."
mathjax: true
toc: true
---
## DBSCAN Introduction
- Density-based spatial clustering of applications with noise (DBSCAN)
- Summary: DBSCAN is a density-based clustering method that discovers clusters of nonspherical shape.
- Main Concept: Locate regions of high density that are separated from one another by regions of low density.
- It also marks as outliers the points that are in low-density regions.  
- Implicit assumptions about the method:
    - Densities across all the clusters are the same.
    - Cluster sizes or standard deviations are the same.  


- __Density of region__:  
    Mainly defined by 2 parameters
    - Density at a point P: Number of points within a circle of Radius $\epsilon$ from point P.
    - Dense Region: For each point in the cluster, a circle with radius $\epsilon$ contains at least minimum number of points ($MinPts$)  
    
    
- `The Epsilon neighborhood` of a point P in the database D is defined as $N(p) = \\{q \in D \mid dist(p, q) \leq \epsilon\\}$
  - The $dist$ function is usually defined by `Euclidean Distance`


- __3 classification of points__:
    - `Core point`: if the point has $\mid N(p)\mid \geq MinPts$ 
    - `Border point`: if the point has $\|N(p)\| < MinPts$ but it lies in the neighborhood of another `Core point`.
    - `Noise`: any data point that is __neither__ `Core` nor `Border` point  
    
    
- __Density Reachable/Density Connected/Directly Density Reachable__
    - `Directly Density Reachable`: Data-point $a$ is directly density reachable from a point $b$ if 
        - $b$ is a core point
        - $a$ is in the epsilon neighborhood of $b$
    - `Density Reachable`: Data-point $a$ is density reachable from a point $b$ if 
        - For a chain of points $b_1, b_2, ...b_n$, $b_1 = b, b_n = a$, $b_{i+1}$ is directly density reachable from $b_i$.
        - `Density reachable` is transitive in nature but, just like `direct density reachable`, it is not symmetric
    - `Density Connected`:  Data-point $a$ is density connected to a point $b$ if 
        - with respect to $\epsilon$ and $MinPts$ there is a point $c$ such that, both $a$ and $b$ are `density reachable` from $c$ w.r.t. to $\epsilon$ and $MinPts$


- __Procedure__
    1. Starts with an arbitrary point which has not been visited and its neighborhood information is retrieved from the $\epsilon$ parameter.
    2. If this point contains $\geq MinPts$ neighborhood points, cluster formation starts. 
      Otherwise the point is labeled as `noise`. 
      - This point can be later found within the $\epsilon$ neighborhood of a different point and, thus can be made a part of the cluster.
    3. If a point is found to be a core point then the points within the $\epsilon$ neighborhood is also part of the cluster. So all the points found within $\epsilon$ neighborhood are added, along with their own $\epsilon$ neighborhood, if they are also core points.
    4. Continue the steps above (1-3) until the `density-connected` cluster is completely found.
    5. The process restarts with a new point which can be a part of a new cluster or labeled as noise.

#### 2. Pros & Cons
**Pros**
- Identifies randomly shaped clusters
- doesn’t necessitate to know the number of clusters in the data previously (as opposed to K-means)
- Handles noise

**Cons**
- If the database has data points that form clusters of __varying density__, then DBSCAN fails to cluster the data points well, since the clustering depends on ϵ and MinPts parameter, they cannot be chosen separately for all clusters
    - May Overcome this issue by running additional rounds over large clusters
- If the data and features are not so well understood by a domain expert then, setting up $\epsilon$ and $MinPts$ could be tricky
- Computational complexity — when the dimensionality is high, it takes $O(n^2)$

#### 3. Code Implementation
{% codeblock lang:python%}
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# To choose the best combination of the algorithm parameters I will first create a matrix of investigated combinations.
from itertools import product

mall_data = pd.read_csv('Mall_Customers.csv')
X_numerics = mall_data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']] # subset with numeric variables only

eps_values = np.arange(8,12.75,0.25) # eps values to be investigated
min_samples = np.arange(3,10) # min_samples values to be investigated
DBSCAN_params = list(product(eps_values, min_samples))

no_of_clusters = []
sil_score = []

for p in DBSCAN_params:
    DBS_clustering = DBSCAN(eps=p[0], min_samples=p[1]).fit(X_numerics)
    no_of_clusters.append(len(np.unique(DBS_clustering.labels_)))
    sil_score.append(silhouette_score(X_numerics, DBS_clustering.labels_))


tmp = pd.DataFrame.from_records(DBSCAN_params, columns =['Eps', 'Min_samples'])   
tmp['No_of_clusters'] = no_of_clusters

pivot_1 = pd.pivot_table(tmp, values='No_of_clusters', index='Min_samples', columns='Eps')

fig, ax = plt.subplots(figsize=(12,6))
sns.heatmap(pivot_1, annot=True,annot_kws={"size": 16}, cmap="YlGnBu", ax=ax)
ax.set_title('Number of clusters')
plt.show()

tmp = pd.DataFrame.from_records(DBSCAN_params, columns =['Eps', 'Min_samples'])   
tmp['Sil_score'] = sil_score

pivot_1 = pd.pivot_table(tmp, values='Sil_score', index='Min_samples', columns='Eps')

fig, ax = plt.subplots(figsize=(18,6))
sns.heatmap(pivot_1, annot=True, annot_kws={"size": 10}, cmap="YlGnBu", ax=ax)
plt.show()

DBS_clustering = DBSCAN(eps=12.5, min_samples=4).fit(X_numerics)

DBSCAN_clustered = X_numerics.copy()
DBSCAN_clustered.loc[:,'Cluster'] = DBS_clustering.labels_ # append labels to points


DBSCAN_clust_sizes = DBSCAN_clustered.groupby('Cluster').size().to_frame()
DBSCAN_clust_sizes.columns = ["DBSCAN_size"]
display(DBSCAN_clust_sizes)


outliers = DBSCAN_clustered[DBSCAN_clustered['Cluster']==-1]

fig2, (axes) = plt.subplots(1,2,figsize=(12,5))


sns.scatterplot('Annual Income (k$)', 'Spending Score (1-100)',
                data=DBSCAN_clustered[DBSCAN_clustered['Cluster']!=-1],
                hue='Cluster', ax=axes[0], palette='Set1', legend='full', s=45)

sns.scatterplot('Age', 'Spending Score (1-100)',
                data=DBSCAN_clustered[DBSCAN_clustered['Cluster']!=-1],
                hue='Cluster', palette='Set1', ax=axes[1], legend='full', s=45)

axes[0].scatter(outliers['Annual Income (k$)'], outliers['Spending Score (1-100)'], s=5, label='outliers', c="k")
axes[1].scatter(outliers['Age'], outliers['Spending Score (1-100)'], s=5, label='outliers', c="k")
axes[0].legend()
axes[1].legend()
plt.setp(axes[0].get_legend().get_texts(), fontsize='10')
plt.setp(axes[1].get_legend().get_texts(), fontsize='10')

plt.show()
{% endcodeblock %}