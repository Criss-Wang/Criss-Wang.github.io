---
title: "Recommender Systems: I. Content-Based Filtering And Collaborative Filtering"
date: 2020-04-01
layout: single
author_profile: true
categories:
  - Recommender Systems
tags: 
  - Content-Based Filtering
  - Collaborative Filtering
excerpt: "Welcome to the world the recommenders"
mathjax: "true"
---
## Overview
The rapid growth of data collection has led to a new era of information. Data is being used to create more efficient systems and this is where Recommendation Systems come into play.  Recommendation Systems are a type of **information filtering systems** as they improve the quality of search results and provides items that are more relevant to the search item or are realted to the search history of the user. They are used to predict the **rating** or **preference** that a user would give to an item. Almost every major tech company has applied them in some form or the other: Amazon uses it to suggest products to customers, YouTube uses it to decide which video to play next on autoplay, and Facebook uses it to recommend pages to like and people to follow. Moreover,  companies like Netflix and Spotify  depend highly on the effectiveness of their recommendation engines for their business and success.
## Traditional recommender system models
There are basically three types of traditional recommender systems, let's use the example of movie recommendation (e.g. Netflix):

- **Demographic Filtering**: They offer generalized recommendations to every user, based on movie popularity and/or genre. The System recommends the same movies to users with similar demographic features. Since each user is different , this approach is considered to be too simple. The basic idea behind this system is that movies that are more popular and critically acclaimed will have a higher probability of being liked by the average audience.
- **Content Based Filtering**: They suggest similar items based on a particular item. This system uses item metadata, such as genre, director, description, actors, etc. for movies, to make these recommendations. The general idea behind these recommender systems is that if a person liked a particular item, he or she will also like an item that is similar to it.
- **Collaborative Filtering**: This system matches persons with similar interests and provides recommendations based on this matching. Collaborative filters do not require item metadata like its content-based counterparts.

In later blogs, we will talk about more recent models for recommender systems, including factorization machines and deep learning based models.

## Content Based Filtering
### 1. Definition
- Use additional information about users and/or items.
    - Example: User features: age, the sex, the job or any other personal information
    - Exmaple: Item features: the category, the main actors, the duration or other characteristics for the movies.
- Main idea: given the set of features (both User and Item), apply a method to identify the model that explain the observed user-item interactions.  <br>
  <img src="/images/Machine%20learning/Content_flow.png" alt="Content_flow" width="500px" align="center" />
- Little concern about "Cold Start": new users or items can be described by their characteristics (content) and so relevant suggestions can be done for these new entities
- One key tool used: `Term Frequency-Inverse Document Frequency (TF-IDF)`:
    - TF: the frequency of a word in a document
    - IDF: the inverse of the document frequency among the whole corpus of documents. 
    - log: log function is taken to dampen the effect of high frequency word (0 vs 100 => 0 vs 2 (log100))
- Note that normalization is needed before we apply `TF-IDF` because the initial feature map are all 1's and 0's, but the __log__ function will remove all these differentiation. In the end the TF score will just be 1/0

### 2. Limitation
- They are not good at capturing inter-dependence or complex behaviours. For example: A user may prefer gaming + tv the most while a pure tv is not really his favourate. 

### 3. Code Sample
```python
import numpy as np # linear algebra
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

books = pd.read_csv('goodread/books.csv', encoding = "ISO-8859-1")
ratings = pd.read_csv('goodread/ratings.csv', encoding = "ISO-8859-1")
book_tags = pd.read_csv('goodread/book_tags.csv', encoding = "ISO-8859-1")
tags = pd.read_csv('goodread/tags.csv')

tags_join_DF = pd.merge(book_tags, tags, left_on='tag_id', right_on='tag_id', how='inner')
to_read = pd.read_csv('goodread/to_read.csv')

tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(books['authors'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

titles = books['title']
indices = pd.Series(books.index, index=books['title'])

# Function that get book recommendations based on the cosine similarity score of book authors
def authors_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21]
    book_indices = [i[0] for i in sim_scores]
    return titles.iloc[book_indices]

books_with_tags = pd.merge(books, tags_join_DF, left_on='book_id', right_on='goodreads_book_id', how='inner')

tf1 = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix1 = tf1.fit_transform(books_with_tags['tag_name'].head(10000))
cosine_sim1 = linear_kernel(tfidf_matrix1, tfidf_matrix1)

# Build a 1-dimensional array with book titles
titles1 = books['title']
indices1 = pd.Series(books.index, index=books['title'])

# Function that get book recommendations based on the cosine similarity score of books tags
def tags_recommendations(title):
    idx = indices1[title]
    sim_scores = list(enumerate(cosine_sim1[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21]
    book_indices = [i[0] for i in sim_scores]
    return titles.iloc[book_indices]

temp_df = books_with_tags.groupby('book_id')['tag_name'].apply(' '.join).reset_index()

books = pd.merge(books, temp_df, left_on='book_id', right_on='book_id', how='inner')

books['corpus'] = (pd.Series(books[['authors', 'tag_name']]
                .fillna('')
                .values.tolist()
                ).str.join(' '))

tf_corpus = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix_corpus = tf_corpus.fit_transform(books['corpus'])
cosine_sim_corpus = linear_kernel(tfidf_matrix_corpus, tfidf_matrix_corpus)

# Build a 1-dimensional array with book titles
titles = books['title']
indices = pd.Series(books.index, index=books['title'])

# Function that get book recommendations based on the cosine similarity score of books tags
def corpus_recommendations(title):
    idx = indices1[title]
    sim_scores = list(enumerate(cosine_sim_corpus[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21]
    book_indices = [i[0] for i in sim_scores]
    return titles.iloc[book_indices]

corpus_recommendations("The Hobbit")

corpus_recommendations("Twilight (Twilight, #1)")
```

## Collaborative Filtering
### 1. Definition
- Based solely on the past interactions recorded between users and items in order to produce new recommendations
- Main idea: Past user-item interactions are sufficient to detect similar users and/or similar items and make predictions based on these estimated proximities.
- Every user and item is described by a feature vector or embedding. It creates embedding for both users and items on its own. It embeds both users and items in the same embedding space.
- 2 Major Types:
    - Memory Based
        - Users and items are represented directly by their past interactions (large sparce vector)
        - Recommendations are done following nearest neighbour information
        - No latent model is assumed
        - Theoretically a low bias but a high variance.
        - Usualy recommend those items with high rating for a user $x_u$: $R_{x_u} = \frac{\sum_{i=0}^{n}R_iW_i}{\sum_{i = 0}^{n}W_i}$
            - $R_{x_u}$: the rating given to $x$ by user $u$ 
            - $i = 0 - i = n$: users similar to $u$ / items similar to $x$
            - $R_i$: respective ratings
            - $W_i$: similarity score for $i$-th item/user similar to $x$/$u$ (deduced using the similarity metircs shown below)
        - Similarity Metrics:
            - `Cosine Similarity`
            - `Dot Product`
            - `Euclidean distance`
            - `Pearson Similarity`: $r = \frac{\sum_{i = 1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i = 1}^{n}(x_i - \bar{x})^2\sum_{i = 1}^{n}(y_i - \bar{y})^2}}$
        - __Limitations__:
            1. Don't scale easily
            2. KNN algorithm has a complexity of O(ndk) 
            3. Users may easily fall into a "information confinement area" which only give too precise/general information
        - __Overcome Limitation__: Use Approximate nearest neighbour (ANN) or take advantage of sparse matrix
        - 2 Types:
            - __User-User__:
                - Identify users with the most similar “interactions profile” (nearest neighbours) in order to suggest items that are the most popular among these neighbours (and that are “new” to our user). 
                - We consider that two users are similar if they have interacted with a lot of common items in the same way (similar rating, similar time hovering…). => Prevents overfitting
                - As, in general, every user have only interacted with a few items, it makes the method pretty sensitive to any recorded interactions (__high variance__)
                - Only based on interactions recorded for users similar to our user of interest, we obtain more personalized results (__low bias__)
            - __Item-Item__:
                - Find items similar to the ones the user already “positively” interacted with
                - Two items are considered to be similar if most of the users that have interacted with both of them did it in a similar way. 
                - A lot of users have interacted with an item, the neighbourhood search is far less sensitive to single interactions (__lower variance__)
                - Interactions coming from every kind of users are then considered in the recommendation, making the method less personalised (__more biased__)
                - VS __User-User__: Less personalized, but more robust
    - Model Based
        - New reprensentations of users and items are build based on a model (small dense vectors)
        - The model "derives" the relevant features of the user-item interactions
        - Recommendations are done following the model information
        - May contain interpretability issue
        - Theoretically a higher bias but a lower variance
        - 3 Types:
            - __Clustering__
                - Simple KNN/ANN will do on these metrices
            - __Matrix Factorization__
                - Main assumption: There exists a very low dimensional latent space of features in which we can represent both users and items and such that the interaction between a user and an item can be obtained by computing the dot product of corresponding dense vectors in that space.
                - Generate the factor matrices as feature matrices for users and items.
                - Idea: $\text{Find X, Y s.t. } M_{n \times m} \approx X_{n \times l} \cdot Y_{m \times l}^T$
                    - $M_{n \times m}$: Interaction matrix of ratings, usually sparse
                    - $X_{n \times l}$: User matrix
                    - $Y_{m \times l}$: Item matrix
                    - $l$: the dimension of the latent space 
                - Advanced Factorization methods:
                    1. __SVD__: 
                        - Not so well due to the sparsity of matrix
                        - $R = UDV^T$: $U$ is item matrix; $V$ is user matrix
                    2. __WMF__ (Weighted Matrix Factorization)
                        - Weight applied to rated/non-rated entries
                        - Similar to __NMF__ but also consider non-rated ones by associating a weight to each entry
                    3. __NMF__:
                        - Uses only the observed or rated one
                        - Performs well with sparse matrices
                        - $Loss = \sum_i \sum_j (X_iY_j^T - M_{i,j})^2\space for \space r(i,j) = 1$ where $r(i,j)$ indicates the $j$-th item rated by $i$-th user
                - Minimizing the objective function
                    - Most common: `Weighted Alternating Least Squares`
                    - Formula: $ (X,Y) = \arg \min\limits_{X,Y} \frac{1}{2}\sum\limits_{(i,j) \in E}[(X_i)(Y_j)^T - M_{ij}]^2 + \frac{\lambda}{2}(\sum\limits_{i,k}(X_{ik})^2 + \sum\limits_{j,k}(Y_{jk})^2)$
                    - Regularized minimization of "rating reconstruction error"
                    - Optimization process via Gradient Descent (Reduce runtime by batch running)
                    - Instead of solving for $X$ and $Y$ together, we alternate between the above two equations.
                        - Fixing $X$ and solving for $Y$
                        - Fixing $Y$ and solving for $X$
                    - This algorithm gives us an approximated result (two equations are not convex at the same time -> can’t reach a global minimum -> local minimum close to the global minimum) 
- For a fixed set of users and items, new interactions recorded over time bring new information and make the system more and more effective.
- Solution to "Cold Start" problem:
    1. Heuristics to generate embeddings for fresh items
        1. Recommending random items to new users/recommend new item to random users
        2. Recommending popular items to new usres/recommend new items to most active users
        3. Recomeending a set of various items to new users or a new item to a set of various users
        4. Use a non collaborative method for early life of the user/item
    2. Projection in WALS (given current optimal $X$ and $Y$)

### 2. Pros & Cons
**Pros**
- Require no information about users or items (more versatile)

**Cons**
- Cold Start problem: 
    1. Impossible to recommend anything to new users or to recommend a new item to any users
    2. Many users or items have too few interactions to be efficiently handled.

### 3. Comparison with Content Based Method
1. Content based methods suffer far less from the cold start problem than collaborative approaches
2. CB is much more constrained (because representation of users and/or items are given)
3. CB tends to have the highest bias but the lowest variance

### 4. Code samples
```python
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
import os
from textwrap import wrap

# Read the input training data
input_data_file_movie = "movie.csv"
input_data_file_rating = "rating.csv"

movie_data_all = pd.read_csv(input_data_file_movie)
rating_data_all = pd.read_csv(input_data_file_rating)

# Keep only required columns
movie_data_all = movie_data_all.drop(['genres'], axis=1)
rating_data_all = rating_data_all.drop(['timestamp'], axis=1)

# Pick all ratings
#num_ratings = 2000000
rating_data = rating_data_all.iloc[:, :]
movie_rating_merged_data = movie_data.merge(rating_data, on='movieId', how='inner')
movie_rating_merged_pivot = pd.pivot_table(movie_rating_merged_data,
                                           index=['title'],
                                           columns=['userId'],
                                           values=['rating'],
                                           dropna=False,
                                           fill_value=0
                                          )

# Create a matrix R, such that, R(i,j) = 1 iff User j has selected a rating for Movie i. R(i,j) = 0 otherwise.
R = np.ones(Y.shape)
no_rating_idx = np.where(Y == 0.0)

# Assign n_m (number of movies), n_u (number of users) and n_f (number of features)

n_u = Y.shape[1]
n_m = Y.shape[0]
n_f = 2  # Because we want to cluster movies into 2 genres

# Setting random seed to reproduce results later
np.random.seed(7)
Initial_X = np.random.rand(n_m, n_f)
Initial_Theta = np.random.rand(n_u, n_f)

# Cost Function
def collabFilterCostFunction(X, Theta, Y, R, reg_lambda):
    cost = 0
    error = (np.dot(X, Theta.T) - Y) * R
    error_sq = np.power(error, 2)
    cost = np.sum(np.sum(error_sq)) / 2
    cost = cost + ((reg_lambda/2) * ( np.sum(np.sum((np.power(X, 2)))) + np.sum(np.sum((np.power(Theta, 2))))))
    return cost

# Gradient Descent
def collabFilterGradientDescent(X, Theta, Y, R, alpha, reg_lambda, num_iters):
    cost_history = np.zeros([num_iters, 1])
    
    for i in range(num_iters):
        error = (np.dot(X, Theta.T) - Y) * R
        X_grad = np.dot(error, Theta) + reg_lambda * X
        Theta_grad = np.dot(error.T, X) + reg_lambda * Theta
        
        X = X - alpha * X_grad 
        Theta = Theta - alpha * Theta_grad
        
        cost_history[i] = collabFilterCostFunction(X, Theta, Y, R, reg_lambda)
        
    return X, Theta, cost_history

# Tune hyperparameters
alpha = 0.0001
num_iters = 100000
reg_lambda = 1

# Perform gradient descent to find optimal parameters
X, Theta = Initial_X, Initial_Theta
X, Theta, cost_history = collabFilterGradientDescent(X, Theta, Y, R, alpha, reg_lambda, num_iters)
cost = collabFilterCostFunction(X, Theta, Y, R, reg_lambda)
print("Final cost =", cost)

user_idx = np.random.randint(n_u)
pred_rating = []
print("Original rating of an user:\n", Y.iloc[:,user_idx].sort_values(ascending=False))

predicted_ratings = np.dot(X, Theta.T)
predicted_ratings = sorted(zip(predicted_ratings[:,user_idx], Y.index), reverse=True)
print("\nPredicted rating of the same user:")
_ = [print(rating, movie) for rating, movie in predicted_ratings]
``` 