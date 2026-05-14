---
title: "Recommender Systems: II. Factorization Machine"
excerpt: "Matrix factorization will save us from sparsity"
layout: post
date: 2020/04/04
updated: 2022/3/29
categories:
  - Blogs
tags: 
  - Linear Algebra
  - Recommender Systems
mathjax: true
toc: true
---
### Factorization Machine
#### 1. Definition
- In essence, a generalized `matrix factorization` method
- `Field`: A type/column in the original dataset
- `Feature`: A value in the Field (Nike is a feature, Brand is a field)
- Movitation:
   - Traditional regression methods cannot handle sparse matrix very well (too much waste in computation time on null values)
   - FM solves the problem of considering pairwise feature interactions (linear time complexity). 
   - It allows us to train, based on reliable information (latent features) from every pairwise combination of features in the model.
- Main logic:
    - Instead of using field as column, each feature has a column.
    - So the columns are basically one-hot-encoding for each value in the field and the row is user id
    - each row covers all the information a user has
    - log-loss function to minimize: $f(x) = w_0 + \sum_{p = 1}^{P}w_px_p + \sum_{p = 1}^{P-1}\sum_{q=p+1}^{P}<v_p,v_q>x_px_q$
        - `w_i`: feature parameter vector (to be optimized)
        - `x_i`: feature vector (column, given)
        - `v_i`: latent vector of predefined low dimension k  (to be optimized)
    - The idea here is that except for individual feature, it consider the combination of 2 features (hence a degree = 2) as a factor
- Extension: `Field-aware FM`
    - For each feature, the parameter vector is no longer unique
    - A feature may interact with other features with different fields. Hence we differentiate the parameter vector for a feature based on the field of its interacting feature
    - E.g: $w_{Gaming, Gender}*w_{Male,Activity}$ $\implies$ Gaming is an activity, Make is a gender; The parameter vector for Male may also be a $w_{Male, Brand}$ if it is interacting with a brand like Nike
- Important note on numerical features
    - Numerical features either need to be discretized (transformed to categorical features by breaking the entire range of a particular numerical feature into smaller ranges and label encoding each range separately).
    - Another possibility is to add a dummy field which is the same as feature value will be numeric feature for that particular row (For example a feature with value 45.3 can be transformed to 1:1:45.3). However, the dummy fields may not be informative because they are merely duplicates of features.

#### 2. Code Sample
- Note that the code below will faill because we haven't installed the xlearn package (too tedious)
- Refer to the code to get an inspiration
- Only apply the code if you have the need to use FM or FFM in your model
- Note that usually DL method works better for the FM-integrated recommender

```python
import pandas as pd
import xlearn as xl
train = pd.read_csv('loan prediction/train_u6lujuX_CVtuZ9i.csv')
import warnings
warnings.filterwarnings('ignore')

cols = ['Education','ApplicantIncome','Loan_Status','Credit_History']
train_sub = train[cols]
train_sub['Credit_History'].fillna(0, inplace = True)
dict_ls = {'Y':1, 'N':0}
train_sub['Loan_Status'].replace(dict_ls, inplace = True)

## train-test split
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(train_sub, test_size = 0.3, random_state = 5)
```
Next, we need to convert the dataset to libffm format which is necessary for xLearn to fit the model. Following function does the job of converting dataset in standard dataframe format to libffm format. `df = Dataframe` to be converted to ffm format
- Type = \'Train\' / \'Test\'/ \'Valid\'
- Numerics = list of all numeric fields
- Categories = list of all categorical fields
- Features = list of all features except the Label and Id

```python
def convert_to_ffm(df,type,numerics,categories,features):
    currentcode = len(numerics)
    catdict = {}
    catcodes = {}
    
    # Flagging categorical and numerical fields
    for x in numerics:
         catdict[x] = 0
    for x in categories:
         catdict[x] = 1
    
    nrows = df.shape[0]
    ncolumns = len(features)
    with open(str(type) + "_ffm.txt", "w") as text_file:
    
    # Looping over rows to convert each row to libffm format
        for n, r in enumerate(range(nrows)):
             datastring = ""
             datarow = df.iloc[r].to_dict()
             datastring += str(int(datarow['Loan_Status'])) # Set Target Variable here
             
            # For numerical fields, we are creating a dummy field here
             for i, x in enumerate(catdict.keys()):
                 if(catdict[x]==0):
                     datastring = datastring + " "+str(i)+":"+ str(i)+":"+ str(datarow[x])
                 else:
            
            # For a new field appearing in a training example
                     if(x not in catcodes):
                         catcodes[x] = {}
                         currentcode +=1
                         catcodes[x][datarow[x]] = currentcode #encoding the feature
             
            # For already encoded fields
                     elif(datarow[x] not in catcodes[x]):
                         currentcode +=1
                         catcodes[x][datarow[x]] = currentcode #encoding the feature
                     
                     code = catcodes[x][datarow[x]]
                     datastring = datastring + " "+str(i)+":"+ str(int(code))+":1"

             datastring += '\n'
             text_file.write(datastring)
```

the <kbd>xLearn</kbd> library can handle csv as well as libsvm format for implementation of FMs while we necessarily need to convert it to libffm format for using FFM. Once we have the dataset in libffm format, we could train the model using the <kbd>xLearn</kbd> library. <kbd>xLearn</kbd> can automatically performs early stopping using the validation/test logloss and we can also declare another metric and monitor on the validation set for each iteration of the stochastic gradient descent.

```python
ffm_model = xl.create_ffm()

ffm_model.setTrain("train_ffm.txt")

param = {'task':'binary', 
         'lr':0.2,
         'lambda':0.002, 
         'metric':'acc'}

# Start to train
# The trained model will be stored in model.out
ffm_model.fit(param, './model.out')

# The library also allows us to use cross-validation using the cv() function:
ffm_model.cv(param)

# Prediction task
ffm_model.setTest("test_ffm.txt") # Test data
ffm_model.setSigmoid() # Convert output to 0-1

# Start to predict
# The output result will be stored in output.txt
ffm_model.predict("./model.out", "./output.txt")
```