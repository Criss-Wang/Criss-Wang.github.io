---
title: "The Data Mining Triology: I. Preparation"
excerpt: "Know how to get data from various sources and load them successfully"
date: 2019/08/25
updated: 2022/8/19
categories:
  - Blogs
tags: 
  - Data Mining/Data Engineering
layout: post
mathjax: true
toc: true
---
### Overview
In this Data Mining triology, I\'m going to present the following critical steps each scientist should perform when handling the data:
1. Data Preparation
  Load and integrate data sources
2. Data Cleaning
  Prepocessing the data to make them meaningful and usable
3. Exploratory Data Analysis
  Analyze the data and understand its pattern. Make corresponding adjustments to data along the way

In this blog, let\'s talk about the first one -- Data Preparation.
### Sources of data
There are indeed a great variety of data sources since the age of machine learning started. While we may come across a wide variety of data types (image, video, text, sheet, signal... you name it), we often can get these data from some popular website:
1. [Google Datasets Search](https://toolbox.google.com/datasetsearch)
   - Pros:
     - Wide coverage, can find whatever dataset you want
   - Cons:
     - Some datasets are not actually accessible, and the website does not indicate that at all!
2. Government Datasets
   - Pros:
     - Most of them are publically available, which is good
     - Most of these data are well preprocessed, so you don\'t have to worry about it
   - Cons:
     - Sometimes, the size of the dataset is not large enough for meaningful projects
     - Governments may not release the most up-to-date datasets. Hence the effectiveness of prediction models may be questionable
3. [Kaggle Datasets](https://www.kaggle.com/datasets)
   - Pros:
     - Really easy to get the data: via command line. This often preferred by professional engineers. See [tutorial](https://www.kaggle.com/docs/api) on kaggle api
     - Aside from the datasets themselves, you can often find a bunch of enthusiasts on machine learning in Kaggle and excellent tutorials on the datasets you found.
   - Cons:
     - It takes a bit of practice to get along with Kaggle. Passion and drive are the key to success in Kaggle
4. [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
   - Pros:
     - One of the most widely used repository for machine learning datasets
     - Very Often, the datasets are related to academic/industrial research projects, so it is extremely helpful to researchers
   - Cons:
     - As a well-aged repo, the datasetes there certainly have been studied extensively. So it may not be so useful for new breakthroughs (but still, it should be very helpful for beginners)

### Code for loading the data
The most common format for machine learning data is CSV files, and we are using python 3.x here for actual code.
This step should mark the start of your notebook (after `np/pd/sklearn/plt`). 
```python
import pandas as pd
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
```
Note that this is too widely abused that people forget about other ways to load data:
#### Load CSV with Python Standard Library
```python
import csv
import numpy as np
raw_data = open("your filename here", 'rt')
reader = csv.reader(raw_data, delimiter=',')
x = list(reader)
data = np.array(x).astype('float')
```
#### Load CSV with Numpy
```python
import numpy
raw_data = open("your filename here", 'rt')
reader = numpy.loadtxt(raw_data, delimiter=",")
```

#### Load CSV with URL
```python
from numpy import loadtxt
from urllib.request import urlopen
url = 'URL to a dataset'
raw_data = urlopen(url)
dataset = loadtxt(raw_data, delimiter=",")
```
Now be cautioned that this is just for CSV files. There are a lot of other data formats, and Google is always your best friend in finding methods to load datasets.
{: .notice--info .notice--x-large}

### That was input, how about output?
Converting data from python objects into byte streams is known as Pickling or Serialization. This allows your own data to be passed around efficiently. Very often, they are stored as `.pkl` or `.json` files. 


#### Python Pickle and JSON
The following table is inspired by [this tutorial](https://www.educba.com/python-pickle-vs-json/)

|                  | Python Pickle                                        | JSON | 
| --------         | --------                                               | ------    | 
| Definition       | Python Pickle is the process of converting python objects (list, dict, tuples, etc.) into byte streams which can be saved to disks or can be transferred over the network. The byte streams saved on the file contains the necessary information to reconstruct the original python object. The process of converting byte streams back to python objects is called de-serialization.                   | JSON stands for JavaScript Object Notation. Data Stored can be loaded without having the need to recreate the data again.| 
| Storage format   | Binary serialization format                            | Simple text serialization format, human-readable  | 
| Storage Versatility          | Not only data entries, but classes and methods can be serialized and de-serialized | JSON is limited to certain python objects, and it cannot serialize every python object, such as classes and functions| 
| Language dependency| Very reliant on the language (Python specific) and versions (2.x pickle files may not be compatible in 3.x env)           | JSON is supported by almost all programming languages.|
| Speed | Slower serialization and de-serialization in pickle | Lightweights, much faster than pickle|
| Security | There is always security risks with pickle files | JSON is generally secure|

<br/>
With the above table in mind, one can choose their outputs accordingly.

### Conclusion
Loading data is merely the first step and people can quickly learn to apply them. However, I/O choices does matter, and one should be cautious about them. Now, lets step into the second step in data mining: cleaning data.