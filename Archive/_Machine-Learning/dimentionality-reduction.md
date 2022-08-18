---
title: "Dimension Reduction: Life savers"
date: 2020-03-04
layout: single
author_profile: true
categories:
  - Unsupervised Learning
tags: 
  - Matrix Computation
  - Dimensionality Reduction
excerpt: "We must dispel the curse of dimensionality!"
mathjax: "true"
---
## Overview
High dimensional data modeling has always been a popular topic of discussion. Many research and work are done in this field simply because we have limited computational power. Even if quantum technology can greatly boost this power in near future, we will still face the curse of unlimited flow of data and features. Thus, it's actually extremely important that we reduce the amount of data input in a model.

In this blog, we explore several well-known tools for dimensionality reduction, namly Linear Discriminant Analysis (LDA), Principle Component Analysis (PCA) and Nonnegative Matrix Factorization (NMF).
## LDA
### 1. Definition
- Can be either a predictive modeling algorithm for multi-class classification or a dimensionality reduction technique
- A Generative Learning Algorithm based on labeled data
- Assumes Gaussian Distribution for $X\|_{y = k}$ ; Each attribute has the same variance (Mean removal/Feature Engineering with Log/Root functions/Box-Cox transformation needed)
- The class calculated from the discrimination function $D_k(x)$ as having the largest value will be the output classification ($y$)
- LDA creates a new axis based on 
     1. Maximize the distance between means
     2. Minimize the variations within each categories
- __Procedure__   
    - __For Dimensionality Reduction__ (reduced-rank LDA)
        - Compute the d-dimensional mean vectors for the different classes from the dataset.
        - Compute the scatter matrices (in-between-class and within-class scatter matrix).
        - Compute the eigenvectors (e1,e2,...,ed) and corresponding eigenvalues ($\lambda_1, \lambda_2, \ldots, \lambda_d$) for the scatter matrices.
        - Sort the eigenvectors by decreasing eigenvalues and choose k eigenvectors with the largest eigenvalues to form a d×k dimensional matrix W (where every column represents an eigenvector).
        - Use this d×k eigenvector matrix to transform the samples onto the new subspace. This can be summarized by the matrix multiplication: Y=XW (where X is a n×d-dimensional matrix representing the n samples, and y are the transformed n×k-dimensional samples in the new subspace).  
    - __For classification__ (Essentially, LDA classifies the sphered data to the closest class mean.)
        - Perform eigen-decomposition on the pooled covariance matrix $\Sigma = UDU^T$
        - Spheres the data: $X^* \leftarrow D^{-\frac{1}{2}}U^TX$ to produce an identity covariance matrix in the transformed space
        - Obtain group means in the transformed space: $\hat{\mu_1},...,\hat{\mu_k}$
        - Classify $x$ according to $\delta_i(X^*)$:
            
            $y_{x_i} = \arg\max \limits_{i} {x^*}^T\hat{\mu_i} - \frac{1}{2}\hat{\mu_i}^T\hat{\mu_i} + \log\hat{\pi_i}$
            {: style="text-align: center;"}
           where $\hat{\pi_i}$ is the group's prior probability
- Extensions to LDA
    - __Quadratic Discriminant Analysis (QDA)__: Each class uses its own estimate of variance (or covariance when there are multiple input variables).
    - __Flexible Discriminant Analysis (FDA)__: Where non-linear combinations of inputs is used such as splines.
    - __Regularized Discriminant Analysis (RDA)__: Introduces regularization into the estimate of the variance (actually covariance), moderating the influence of different variables on LDA.

### 2. Pros & Cons
**Pros**
- Need Less Data
- Simple prototype classifier: Distance to the class mean is used, it’s simple to interpret.
- The decision boundary is linear: It’s simple to implement and the classification is robust.

**Cons**
- Linear decision boundaries may not adequately separate the classes. Support for more general boundaries is desired.
- In a high-dimensional setting, LDA uses too many parameters. A regularized version of LDA is desired.
- Support for more complex prototype classification is desired.

### 3. Application
- __Bankruptcy prediction__
- __Facial recognition__

### 4. Code implementation
```python
from sklearn.datasets import load_wine
import pandas as pd
import numpy as np
np.set_printoptions(precision=4)
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Loading Data
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Categorical.from_codes(wine.target, wine.target_names)
# Merge X and y (Training set)
df = X.join(pd.Series(y, name='class'))

## The Simply Way: using sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
X_lda = lda.fit_transform(X, y)

## The hard way: Understand the logic here
# Compute means for each class
class_feature_means = pd.DataFrame(columns=wine.target_names)
for c, rows in df.groupby('class'):
    class_feature_means[c] = rows.mean()
class_feature_means

# Compute the Within Class Scatter Matrix using the Mean Vector
within_class_scatter_matrix = np.zeros((13,13))
for c, rows in df.groupby('class'):
    rows = rows.drop(['class'], axis=1)
    
s = np.zeros((13,13))
for index, row in rows.iterrows():
        x, mc = row.values.reshape(13,1), class_feature_means[c].values.reshape(13,1)
        
        s += (x - mc).dot((x - mc).T)
    
        within_class_scatter_matrix += s

# Compute the Between Class Scatter Matrix:
feature_means = df.mean()
between_class_scatter_matrix = np.zeros((13,13))
for c in class_feature_means:    
    n = len(df.loc[df['class'] == c].index)
    
    mc, m = class_feature_means[c].values.reshape(13,1), feature_means.values.reshape(13,1)
    
    between_class_scatter_matrix += n * (mc - m).dot((mc - m).T)

# Compute the Eigenvalues & Eigenvectors, then sort accordingly

eigen_values, eigen_vectors = np.linalg.eig(np.linalg.inv(within_class_scatter_matrix).dot(between_class_scatter_matrix))

pairs = [(np.abs(eigen_values[i]), eigen_vectors[:,i]) for i in range(len(eigen_values))]
pairs = sorted(pairs, key=lambda x: x[0], reverse=True)

# Print the Explained Variance
eigen_value_sums = sum(eigen_values)
print('Explained Variance')
for i, pair in enumerate(pairs):
    print('Eigenvector {}: {}'.format(i, (pair[0]/eigen_value_sums).real))


# Identify the Principle Eigenvalues (here k = 2); Compute the new feature space X_lda
w_matrix = np.hstack((pairs[0][1].reshape(13,1), pairs[1][1].reshape(13,1))).real
X_lda = np.array(X.dot(w_matrix))

le = LabelEncoder()
y = le.fit_transform(df['class']) # Here the y is just the encoded label set

# Visualize
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.scatter(
    X_lda[:,0],
    X_lda[:,1],
    c=y,
    cmap='rainbow',
    alpha=0.7,
    edgecolors='b'
)
```

## PCA
### 1. Definition
- A linear model
- Basic intuition: projecting data onto its orthogonal feature subspace
- It is a technique for feature extraction — it combines our input variables in a specific way, then we can drop the “least important” variables while still retaining the most valuable parts of all of the variables (__high variance__, __independent__, __few number__)
- Each of the “new” variables after PCA are all independent of one another (due to the _linear model_ assumption)
- PCA effectively minimizes error orthogonal to the model itself
- It can only be applied to datasets which are linearly separable
- Complete procedure
    1. Standardize the data.
    2. Obtain the Eigenvectors and Eigenvalues from the covariance matrix or correlation matrix (by performing eigendecomposition), or perform Singular Value Decomposition (SVD) [__SVD preferred due to efficiency and automatic eigenvalue sorting__]
        - Find the hyperplane that accounts for most of the variation using SVD (that hyperplane represents 1st componet);
            - This basically means it tries to minimize the points' distance to the hyperplane/maximize the distance from the projected point to the origin
        - Find the orthogonal subspace, get from the subspace a best hyperplane with largest variation using SVD again (2nd component which is clearly independent of 1st component)
        - Repeat to find all the components
        - Use the eigenvalues to determine the proportion of variation that each component account for;
        - construct the new system using the principle (top k) components; plot the samples using the projections on components as coordinates
    3. We should keep the component if it contributes substantially to the variation
    4. It eventually cluster samples that are highly correlated;


- It is possible to restore all the samples if each component correspond to one distinct variable (Note that # of PC < # of Samples)
- Warning: __scaling__ and __centering__ data is very Important!!!


- __Kernel PCA__
    - an extension of PCA into non-linear dataset by project dataset into a higher dimensional feature space using the `kernel trick` (recall SVM)
    - Some popular kernels are
        - Gaussian/RBF
        - Polynomial
        - Hyperbolic tangent: $K(\vec{x}, \vec{x}') = tanh(\vec{x} \cdot \vec{x}' + \delta)$
    - Note that the kernel matrix still need to be _normalized_ for PCA to use
    - kernel PCA so kernel PCA will have difficulties if we have lots of data points.
    
- __Robust PCA__
    - an extension of PCA to deal with sparsity in the matrix
    - It factorizes a matrix into the sum of 2 matrices, $M = L + S$, where $M$ is the original matrix, $L$ is the low-rank (with lots of redundant information) matrix and $S$ is a sparse matrix (In the case of corrupted data, $S$ often captures the corrupted entries)
    - Application: Latent semantic indexing => $L$ captures all common words while $S$ captures all key words that best identify each document.
    - The minimization is over $\Vert L\Vert_* + \lambda \Vert S\Vert_1$ subject to $M = L + S$. Minimizing L1-norm results in sparse values, while minimizing nuclear norm (sometimes also use Frobenious norm $\Vert L\Vert_F$) leads to sparse singular values (hence low rank)

### 2. Pros & Cons
**Pros**
- Removes Correlated Features
- Improves Algorithm Performance
- Reduces Overfitting

**Cons**
- Independent variables become less interpretable
- Data standardization is must before PCA
- Information Loss (if PCs chosen are not sufficient)

### 3. Application
- When interpretability is not an issue, use pca
- When the dimension is too large or you want to identify features that are independent, use pca

### 4. Comparison
_PCA vs LDA_
- Not as good as LDA in clustering/classification effect, yet idea for Factor analysis
- PCA projects the entire dataset onto a different feature (sub)space, and LDA tries to determine a suitable feature (sub)space in order to distinguish between patterns that belong to different classes


_PCA vs ICA_
- In PCA the basis you want to find is the one that best explains the variability of your data; In ICA the basis you want to find is the one in which each vector is an independent component of your data (which usually has mixed signals/mixed features)

### 5. Code implementation
```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style='white')
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
from sklearn import decomposition
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D

digits = datasets.load_digits()
X = digits.data
y = digits.target

# f, axes = plt.subplots(5, 2, sharey=True, figsize=(16,6))
plt.figure(figsize=(16, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X[i,:].reshape([8,8]), cmap='gray')
    
pca = decomposition.PCA(n_components=2)
X_reduced = pca.fit_transform(X)

print('Projecting %d-dimensional data to 2D' % X.shape[1])

plt.figure(figsize=(12,10))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, 
            edgecolor='none', alpha=0.7, s=40,
            cmap=plt.cm.get_cmap('nipy_spectral', 10))
plt.colorbar()
plt.title('MNIST. PCA projection')

pca = decomposition.PCA().fit(X)

plt.figure(figsize=(10,7))
plt.plot(np.cumsum(pca.explained_variance_ratio_), color='k', lw=2)
plt.xlabel('Number of components')
plt.ylabel('Total explained variance')
plt.xlim(0, 63)
plt.yticks(np.arange(0, 1.1, 0.1))
plt.axvline(21, c='b')
plt.axhline(0.9, c='r')
plt.show()

```

## NMF

### 1. Definition
- It automatically extracts sparse and meaningful (easily interpretable) features from a set of nonnegative data vectors.
- We basically factorize $X_{m\times n}$ into 2 smaller matrices non-negative $W_{m\times r}$ and $H_{r\times n}$ such that $X \approx WH$ and $r \leq \min(m,n)$ ( low-rank approximate factorizations)
- Interpretation of $W$ and $H$:
    - Basically, we can interpret $x_i$  to be a weighted sum of some components, where each row in $H$ is a component, and each row in $W$ contains the weights of each component
- Idea of the algo:
    - Formalize an objective function and iteratively optimize it
    - A local minima is sufficient for the solution
- Objective function to minimize :
    - `Frobenius norm`: $\Vert X-WH\Vert_{F}^{2} \space w.r.t. \space W,H \space s.t. \space W,H \geq 0$
    - `generalized Kullback-Leibler divergence`: $\sum \limits_{ij} (X_{ij}\log\frac{X_{ij}}{WH_{ij}} - X_{ij} + WH_{ij})$
- Choices of Optimization technique used:
    - Coordinate descent (alternative: gradient descent which fix $W$ and optimize $H$, then fix $H$ and optimize $W$ until tolerance is met)
    - Multiplicative Update
        -$H \leftarrow H \frac{W^T X}{W^T W H}$,     $W \leftarrow W \frac{W^T X}{W H H^T}$
- Method to choose the optimal factorisation rank, $r$: 
    - General guideline:  $(n+m)r <nm$
    - Trial and error, 
    - Estimation using SVD based of the decay of the singular values
    - Insights from experts
- Tricks 
    1. Initialization: uses SVD to compute a rough estimate of the matrices $W$ and $H$ . If the non-negativity condition did not exist, taking the top k singular values and their corresponding vectors would construct the best rank k estimate, measured by the frobenius norm. Since $W$ and $H$ must be non-negative, we must slightly modify the vectors we use. 
    2. Regularization: Since the $W$ represents weights of a component, it may produces weights that are too high/low. The classical way is to use $l_1$ or $l_2$ regularization losses

### 2. Application
- NMF is suited for tasks where the underlying factors can be interpreted as non-negative
- Image processing
- Topic Modeling
- Text mining
- Hyperspectral unmixing

### 3. Code Implementation
```python
from operator import itemgetter
from concurrent.futures import ProcessPoolExecutor 
from time import time
import os 
import gensim
import pandas as pd
import itertools
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer, LabelEncoder

lemmatizer = WordNetLemmatizer()

data_path = 'matrix_factorization_arxiv_query_result.json'
articles_df = pd.read_json(data_path)
articles_df.head()


def stem(text):
    return lemmatizer.lemmatize(text)


def map_parallel(f, iterable, **kwargs):
    with ProcessPoolExecutor() as pool:
        result = pool.map(f, iterable, **kwargs)
    return result


def retrieve_articles(start, chunksize=1000):
    return arxiv.query(
        search_query=search_query,
        start=start,
        max_results=chunksize
    )

def vectorize_text(examples_df, vectorized_column='summary', vectorizer=CountVectorizer):

    vectorizer = vectorizer(min_df=2)
    features = vectorizer.fit_transform(examples_df[vectorized_column])

    le = LabelEncoder()
    ohe = OneHotEncoder()
    labels = le.fit_transform(valid_example_categories).reshape(-1, 1)
    labels_ohe = ohe.fit_transform(labels).todense()
    vectorized_data = {
        'features': features,
        'labels': labels,
        'labels_onehot' : labels_ohe
    }
    return vectorized_data, (vectorizer, ohe, le)


def extract_keywords(text):
    """
    Use gensim's textrank-based approach
    """
    return gensim.summarization.keywords(
        text=stem(text),
        lemmatize=True
    )

def filter_out_small_categories(df, categories, threshold=200):

    class_counts = categories.value_counts()
    too_small_classes = class_counts[class_counts < threshold].index
    too_small_classes

    valid_example_indices = ~categories.isin(too_small_classes)
    valid_examples = df[valid_example_indices]
    valid_example_categories = categories[valid_example_indices]
    
    return valid_examples, valid_example_categories

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % (topic_idx + 1)
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

categories = articles_df['arxiv_primary_category'].apply(itemgetter('term'))

main_categories = categories.apply(lambda s: s.split('.')[0].split('-')[0])

main_categories_counts = main_categories.value_counts(ascending=True)
main_categories_counts.plot.barh()
plt.show()

main_categories_counts[main_categories_counts > 200].plot.barh()
plt.show()

categories.value_counts(ascending=True)[-10:].plot.barh()
plt.show()


articles_df['summary_keywords'] = [extract_keywords(i) for i in articles_df['summary']]

article_keyword_lengths = articles_df['summary_keywords'].apply(lambda kws: len(kws.split('\n')))

valid_examples, valid_example_categories = filter_out_small_categories(articles_df, main_categories)

vectorized_data, (vectorizer, ohe, le) = vectorize_text(
    valid_examples,
    vectorized_column='summary_keywords',
    vectorizer=TfidfVectorizer
)

x_train, x_test, y_train, y_test, y_train_labels, y_test_labels = train_test_split(
    vectorized_data['features'],
    vectorized_data['labels_onehot'],
    vectorized_data['labels'],
    stratify=vectorized_data['labels'],
    test_size=0.2,
    random_state=0
)

nmf = NMF(n_components=5, solver='mu', beta_loss='kullback-leibler')

topics = nmf.fit_transform(x_train)

n_top_words = 10

tfidf_feature_names = vectorizer.get_feature_names()
print_top_words(nmf, tfidf_feature_names, n_top_words)

dominant_topics = topics.argmax(axis=1) + 1
categories = le.inverse_transform(y_train_labels[:,0])
pd.crosstab(dominant_topics, categories)
```
