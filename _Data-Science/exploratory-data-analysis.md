---
title: "The Data Mining Triology: III. Analysis"
date: 2019-09-03
layout: single
author_profile: true
categories:
  - Data Mining
tags: 
  - Exploratory Data Analysis
  - Graph Plotting
excerpt: "Let the plot fly ~"
mathjax: "true"
---
## Overview
Finally we have come to the last part in fundamental data mining. This is where people's analytical power shine through. However, we also highlight some cautions engineers should practice in exploratory analysis. 

While data analysis is fascinating, I feel that building models based on the analysis to facilitate business decisions is even more exciting. This heavily relies on machine learning models and artificial intelligence toolkits. I've also written (and will write more in the future) blogs on these topics. Word of reminder: these models require some level of statistical and mathematical foundations, so it really depends on one's interests in developing these models.

## A general view of the dataset
One can always use an easy trick: `YourDataFrameName.describe()` to show the details about your data entries. This gives very good view of properties of your data. A sample output looks like
```python
>>> df.describe(include='all')  
       categorical  numeric object
count            3      3.0      3
unique           3      NaN      3
top              f      NaN      a
freq             1      NaN      1
mean           NaN      2.0    NaN
std            NaN      1.0    NaN
min            NaN      1.0    NaN
25%            NaN      1.5    NaN
50%            NaN      2.0    NaN
75%            NaN      2.5    NaN
max            NaN      3.0    NaN
```
Next let's look into numerical data's pattern first.
## Numerical data distributions
### Generate comprehensive view for the numericals in Data Set
```python
list(set(df.dtypes.tolist()))
df_num = df.select_dtypes(include = ['float64', 'int64']) # select only numerical data
display(df_num.head()) # output the first 5 entries
df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8) # Give the comprehensive views of all the distribution (in histogram) of the numerical values;
```
**Key steps**:  
(i) From the graphs, find which features have similar distributions;  
(ii) Document the discovery for further investigation;
{: .notice--info .notice--x-large}

### Correlation (correlation is affected by outliers)
Find the strongly correlated values with the output. Call this list of values `golden_features_list`.
```python
df_num_corr = df_num.corr()['SalePrice'][:-1] # -1 because the latest row is SalePrice
golden_features_list = df_num_corr[abs(df_num_corr) > 0.5].sort_values(ascending=False)
print("There is {} strongly correlated values with SalePrice:\n{}".format(len(golden_features_list), golden_features_list))
```

### Correlation (outliers removal)
1. Plot the numerical features and see which ones have very few or explainable outliers
2. Remove the outliers from these features and see which one can have a good correlation without their outliers

```python
for i in range(0, len(df_num.columns), 5):
sns.pairplot(data=df_num,
      x_vars=df_num.columns[i:i+5],
      y_vars=['SalePrice'])
```

**Key steps**:  
(i) Spot any clear outliers, document. Think of outlier's plausibility. Think of whether to remove it & document;  
(ii) Spot any clearly linear/non-linear relationships, document;  
(iii) Spot any distribution with a lot of 0's: do Correlation (0 Removal);
{: .notice--info .notice--x-large}

### Correlation (0 Removal)
Removing all 0's in some columns and generate `golden_features_list` again, see if any new features are added. 
```python
import operator

individual_features_df = []
for i in range(0, len(df_num.columns) - 1): # -1 because the last column is SalePrice
  tmpDf = df_num[[df_num.columns[i], 'SalePrice']]
  tmpDf = tmpDf[tmpDf[df_num.columns[i]] != 0]
  individual_features_df.append(tmpDf)

all_correlations = {feature.columns[0]: feature.corr()['SalePrice'][0] for feature in individual_features_df}
all_correlations = sorted(all_correlations.items(), key=operator.itemgetter(1))
for (key, value) in all_correlations:
  print("{:>15}: {:>15}".format(key, value))
  
golden_features_list = [key for key, value in all_correlations if abs(value) >= 0.5]
print("There is {} strongly correlated values with SalePrice:\n{}".format(len(golden_features_list), golden_features_list))
```
Finally, we can give conclusion with respect to the Numerical data distribution analysis.

## Feature to feature (categorical to categorical) correlation Analysis:
	
1. [Heat map](https://towardsdatascience.com/heatmap-basics-with-pythons-seaborn-fb92ea280a6c) for features:  <br/>
  ```python
    corr = df_num.drop('SalePrice', axis=1).corr() # We already examined SalePrice correlations
    plt.figure(figsize=(12, 10))

    sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.4)], 
        cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
        annot=True, annot_kws={"size": 8}, square=True);
    # this generates the heatmap that display highly related features;
    # Note that this map only displays bidirectional relationships;
    # We cannot conclude much if there are relationships among feature set of size >= 3;
  ```  
  **Steps of Analysis for the Heatmap**:
   - First of all, remove all simple correlations (easy to explain & not that relevant)
   - Next, identify the relationships that are pertinent to the questoin/task
   - Lastly, conclude the features that are similar to be easily combined/ need further investigation/ clearly helpful to the task
   - Document the analysis;
	
## Q --> Q (Quantitative to Quantitative relationship)
	
1. Extract strongly correlated quantitative features
```python
features_to_analyse = [x for x in quantitative_features_list if x in golden_features_list]
features_to_analyse.append('SalePrice')
display(features_to_analyse)	
```
2. plot the distribution:
```python
fig, ax = plt.subplots(round(len(features_to_analyse) / 3), 3, figsize = (18, 12))
for i, ax in enumerate(fig.axes):
  if i < len(features_to_analyse) - 1:
        sns.regplot(x=features_to_analyse[i],y='SalePrice', data=df[features_to_analyse], ax=ax)
```
3. Analysis of the distribution:
- Since Linear Regression is give, we focus on analyzing the spread of the data in each graph

## C --> Q (Categorical to Quantitative relationship)

1. Extract Categorical features
```python
categorical_features = [a for a in quantitative_features_list[:-1] + df.columns.tolist() if (a not in quantitative_features_list[:-1]) or (a not in df.columns.tolist())]
df_categ = df[categorical_features]
df_not_num = df_categ.select_dtypes(include = ['O']) # include the non-numerical features
```
2. Apply Boxplot
```python 
plt.figure(figsize = (12, 6))
ax = sns.boxplot(x='SaleCondition', y='SalePrice', data=df_categ) 
# can replace "SaleCondition" with other features
```
3. Apply Distribution plot
```python
fig, axes = plt.subplots(round(len(df_not_num.columns) / 3), 3, figsize=(12, 30))
for i, ax in enumerate(fig.axes):
  if i < len(df_not_num.columns):
        ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=45)
        sns.countplot(x=df_not_num.columns[i], alpha=0.7, data=df_not_num, ax=ax)
fig.tight_layout()
```
Through these plots, we can see that some categories are predominant for some features such as `Utilities`, `Heating`, `GarageCond`, `Functional`... These features may not be relevant for our predictive model.
  
## Conclusion
The methods above cover a wide range of tools being applied in data analytics. There are definitely many more directions in EDA, and I'll update my discovery every time I find some interesting things.