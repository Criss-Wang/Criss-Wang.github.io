---
title: "Data Science Project: Lifecycle"
date: 2020-06-28
comments: false
related: false
share: false
categories:
  - Project
tags: 
  - Data Science
  - Project management
header:
    overlay_image: "/images/Data Science Concept/H2.png"
    overlay_filter: 0.6
excerpt: "Understanding the flow of data science projects"
mathjax: "true"
---
# Overview of Data Science Project

<!--- Most people started their first data science project in campus life, if not their first job experience. However, few started in a proper manner. It is natural for them to have such projects to have a taste of how various data science concepts and tools gets utilised in a project. However, it is important that we following a standardized project workflow when we actually start our work. Some companies might teach you how to run through a data science project in their own fashion. Most data science teams, however, are less likely to give you a step-by-step guidance on the exact flow of such project. This is especially true in recent years, where the data science teams in each companies have reached a scale enough to tolerate small deviations cause by your ignorance of project lifecycle. However, it would be nice to the team you are working with if you have already mastered this simple yet important concept, as it not only amazes your colleagues and boss, but also makes your tranmission between stages of the project much smoother. --->

A formal and impactful data science project should comprise of 4 major components:
- Business insight
- Data acquisition & analysis
- Data modeling & experimentation
- Deployment & Enhancement

![image-center]({{ site.url }}{{ site.baseurl }}/images/ds lifecycle.png){: .align-center}

## Business Insight
This mostly overlooked section is yet mostly valued by the company as a whole. For many novice data scient lovers like me, how much value does a model brings was not taught in our college modules. We used to care only about result accuracy and efficiency. However, whether a model is a good model is hardly dependent on these two metrics. In fact, an ideal model to a company would be a one that is innovative and profitable. It is an important lesson from my first 2 internships in data science companies that __"start a project only if it brings value to a company"__.

## Data preparation
This section ensures that cleaned/reliable dataset can be extracted from database/real-life to build up the model.

This part is the focus of data engineer's job, and data scientists should also know all the common practices well enought to avoid communication barrier with engineers. A few important areas include:
- __Data source__: Verify data collected are reliable
- __Data pipelining__: Frequency and size of newly captured data being processed
- __Data storage__: 
    - Cloud service vs Local database
    - Data format and access control
- __Data analysis__:
    - Wrangling/Structuring
    - Validation/Cleaning (Preprocessing)
    - Analysis (EDA)
    - Insight report & Visualization

**Note:** Data analysis could be a arduous job to do. In fact, this takes majority of the time of data scientists and analysts to work on this part.
{: .notice--info}

## Data Modeling
You say you want your machine learning skills gets utilized? Here comes the section to achieve your ambition!

So you start to think of your XGBoost, Discriminant Analysis, LSTM...

__BUT Wait!__ If you think you're gonna pour much time into creating a wonderful model with perfect accuracy and efficiency here, you dead wrong! Instead of building a perfect model, companies usually push data scientists to create something that __"can sell"__. They need the model to be productionized and produce visible results that can convince their customers. That's why people with lots of project experiences are favored -- they are familiar with all sorts of models and tuning tools so the final result is out __FAST__. So do more Kaggle competitions, do more research projects to make sure you know what to use quickly in the job!

## Deployment
For software engineers, getting their updates deployed is one of the happiest moment. For data scientist, however, the journey has just begun. 

Model performance becomes much more important at this stage. We need to ensure the quality of our model (accuracy, efficiency, latency, scalability, etc). Scientists have to provide credible metrics for model evaluation and conduct hyperparameter optimization to improve the model further. Interactions with product management and software engineering teams gets more frequent. The team needs to monitor the model's performance and ensures its smooth integration with the actual product. __Optimization__ is the theme here, and it's never a simple task. 

## Summary
Seeing the complete cycle, I have to admit that as a student on his way to discover more data science knowledge, I still have a long way to go. 

My school courses can cover lots of theoretical knowledge in data preparation and modeling, but it is really the internship and project experiences that taught me lessons about the generating business insights and proper deployment techniques. 