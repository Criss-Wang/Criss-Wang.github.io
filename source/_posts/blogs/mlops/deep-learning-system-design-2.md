---
title: "Deep Learning System Design - A Checklist (Part II)"
excerpt: "An Overview of how to design a full-stack Deep Learning System"
date: 2024/02/10
categories:
  - Blogs
tags:
  - Deep Learning
  - System Design
layout: post
mathjax: true
toc: true
---

### A quick recap

In the previous post, checklist part I, we\'ve talked about the early stage of designing a deep learning system. These steps are often of paramount importance when we build some ml projects in our courseworks. At the end of these steps, we often have a ready-to-use model that solves the problem at hand. However, if we really want to make it a product, to benefit thousand of users or the community, a lot of engineering work on the backend still need to be done. This includes:

- Saving the model artifact, wrapping the solution up and deploy it
- Creating endpoints for user to interact with, aka model serving
- Iteratively update the model by monitoring the model performance and system performance, and fix any issue related to the product.

Let\'s go through each of the step one by one.

### Step 5: Packaging and Deployment

Technically speaking, **packaging** a model isn\'t really the right word to describe the process of saving a trained model for usage. When people talk about **packaging** a model, they usually mean storing the trained model somewhere to deploy it for future usage. Thus it is closely related to deployments. Hence, when it comes to saving the model, here\'s the things to look out for:

1. What platform do you use to store the model: local? cloud? edge?
2. What metadata do you need?
   - model hyperparams?
   - dependencies (this can be tricky a lot of times)
   - model json files? (example: hugging face models)
3. how do you do the
4. what\'s the size requirement?
5. can we containerize it? (i.e. building an environment easy for deployment and serving)
6. Is model-versioning done effectively?
7. Does the saved model work perfectly in the infrastructure? (GPU? Memory? Network?)
8. Knowing when to update the model

when deploying the model, several strategies can be considered as well. For example:

- Directly use the existing endpoints from experiment tracking tools (e.g. wandb, kubeflow)
- Setup external APIs (SageMaker, AWS Lambda, AWS ECS)
- Shadow Deployment
- A/B Testing (with bandit method sometimes)
- Canal Deployment

### Step 6: Serving

This is where the endpoint becomes crucial, you need to consider several components

- what is the backend api tool you use
- do you containerize your api server?
- do you make it a distributed system? are concurrency and parallelism available options?
- whether the inference task will be cpu/gpu bound or io bound?
- do you consider batch inference? streaming inference? (latency requirement)
- will message queue become important for communication between api server and model server? (e.g. failure recovery)
- are there ways to easily integrate the metrics from serving to the monitoring tool? (callback functions for example)
- how do you handle the input data? (database management)
- how to save the request/response information for future usage? (example: user feedback collection)
- Is there a way to conduct quick test for serving before user acceptance test?
- Security issues?
- How do you direct traffic to different models and collect results from them? (e.g. paired t-test during shadow deployment)

### Step 7: Monitoring

Don\'t forget to do logging as it is super important. Make it structured with time stamps and severity levels. Some of the objects for the data and model components you should log include:

- Data pipeline events,
- Production data (if possible, include the metadata alongside),
- Model metadata; this includes the model version and configuration details,
- Prediction results from the model,
- Prediction results from shadow tests (challenger models); if applicable to your system,
- Ground truth label (if available),
- General operational performance (that is typical to standard monitoring systems).

Some best practices include:

- For your pipeline, you should be logging runs from scheduled time to start time, end time, job failure errors, the number of runs; all to make an unhealthy pipeline easier to troubleshoot.
- For your models, you should be logging the predictions alongside the ground truth (if available), a unique identifier for predictions (prediction_id), details on a prediction call, the model metadata (version, name, hyperparameters, signature), the time the model was deployed to production.
- For your application, you should be logging the number of requests served by the champion model in production, average latency for every serving.
- For your data, log the version of every preprocessed data for each pipeline run that was successful so that they can meet audited and their lineage can be traced.
- For storing the structure of your logs, consider using a JSON format with an actual structure so they can be easily parsed and searched.
- Consider rotating log files for better management; delete old and unnecessary logs that you\'re sure you won\'t need again for auditing or other reasons.

#### System-related

- Throughput
- Latency
- Endpoint Availability
- System Error Rate (e.g. system overload time, number of failed requests)
- Total number of API calls
- CPU/GPU Utility
- Disk I/O
- Memory Utility
- Dependency Health
- Cloud Infra Health
- Resource Cost

#### Model-related

- Error Rate, Model Drifts
- data drift between training data and request data
- Data Quality Issues
- Outliers Detection & Handling
- Retraining Frequency
- Model Versioniong
- Prediction Metrics
- Model Poisoning Attack Detection
- Explainability
- Audit Trails + Privacy
- User Feedback

### Conclusion

While a deep learning system can "almost" be always built following the checklist I made here, we must stay close to our business objective for the system to be truly useful. In that sense, a close connection to our user would be very important, and things like defensive programming, friendly UI and user feedbacks play super important roles. In future posts, I\'ll talk about some of them. Stay tuned ~

### References

1. [A Comprehensive Guide on How to Monitor Your Models in Production - Neptune.ai](https://neptune.ai/blog/how-to-monitor-your-models-in-production-guide)
