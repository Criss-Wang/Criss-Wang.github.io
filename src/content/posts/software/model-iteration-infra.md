---
title: "Model Iteration Series: Validating Model Infra"
excerpt: "The system and product side of concerns"
date: 2024/07/27
categories:
  - Software
tags:
  - ML Infrastructure
  - Model Development
layout: post
mathjax: true
toc: true
---

## **Intro**

As the step 2 of the model iteration process, testing model compatibility with the existing infrastructure is another step we should take care of. It consists of managing the resources for training, deployment as well as any interactions mechanisms with the service backend logic (e.g. API). We will provide a deep dive into the relevant investigations and adjustments required for the model changes proposed to be ready for QA testing.

Again, if you haven't read my blog that provides the high level insights to the series, I would urge you to kindly go through it to gain more context on this write-up. This blog is for LLM models and its related products/services. The considerations and design decisions required for other ML/AI models may be drastically different.

In the upcoming sections, I will discuss the following topics:

- Distinction between Infra in QA/Prod and Infra in Dev
- Key components to look out for when managing the infra
- Differernt layers of Latency Tests and their purposes
- Cost evaluation strategies and their impacts to the business
- Trade-offs to consider during the evaluation

## **Definitions**

When discussion LLM infrastructure, we must set a clear boundary between the infra toolset used in dev environment vs the ones used in staging or production environment. While the latter strive to be as efficient and robust as possible, the dev infra is more on the exploratory phase to identify potential issues the changes to models/services may cause using the current infra setup. For example, applying [multi-token prediction](https://arxiv.org/abs/2404.19737) may sounds attractive to boost the inference speed, but may be completely incompatible with the current GPU resources, or the framework the serves the inference. Hence, it is the Infra team\'s job to ensure the changes are valid on the current setup, or make adjustments like changing the inference engine or the compute to account for the changes.

The dev infra should also be splitted into two categories: **model-based** and **service-based**. Model-based infra are about model training and serving. They strive to get the best out of the computing resources. Usually this involves adopting the most suitable GPU setup and multi-node/multi-server communication links for each model based on the request rate. Avoiding OOM and throttling are bread and butter of the of Model-based infra taks. In the meantime, service-based infra are more concentrated on achieving efficiecny and robustness of the distributed system for the services which the products are based on. This often involves testing the service latency and explore the right scaling strategy for the services that use LLMs. Although it could be QA team\'s responsibility to test the full product use cases _end-to-end_, it it in my personal philosophy that MLEs owns the services from start to finish. As a result, experienced MLEs would keep service-based infra testing to their own for better accountability. Therefore, usually we have specialized MLOps to handle the service-based tasks.

In the model iteration process, the dev infra needs to test both the system issues these changes may cause on current product, and also the performance expectations of these changes. **Latency** and **Cost** are the two major metrics in my knowledge, as they are ones that infra has the most control over.

In the context of LLM, latency metrics often measure the rate at which services handles a task using the model as its main reasoning engine. Usually the proposed changes should meet a certain latency threshold, or achieve a latency goal such that customers don\'t feel the difference in term of speed as a bottom line, and potentially sense a significant speedup as a result. On the other hand, cost is related to the cost of GPU/API/electricity. A proposed change should strive to at least keep the cost at the current level, or reduce the cost such that it impacts the business and future research explorations.

## **Key Components**

Now let\'s talk about details. When testing against infrastructure for MLOps in an LLM based service, there are several key components to consider:

**Model-based**

1. Compute Resources: LLMs require significant computational power for training and inference. Ensure access to high-performance GPUs or specialized AI accelerators to handle the intensive processing demands. This means runtime acceleration and GPU optimization methods need to align with the change proposed.
2. Model Architecture: When a services is utilizing multiple models, or multiple adapters attached to the same foundation model, the structure of the model and its inference strategy become a signifant concern here. Any distillation or model compression conducted at serving time should also be investigated.

**Service-based**

1. Scalable Infrastructure: Implement a flexible and scalable infrastructure that can adapt to changing workloads. Cloud-based solutions or hybrid setups can provide the necessary elasticity. However, introducing a change can potentially break the setup easily. Tests should be ran to prevent such danger.
2. Data management: Are we using caching? Are we deploying across thousands of GPU nodes? How to do failure recovery using logs? Managing these data are critical and any changes introduced should be validated against any infra assumptions on data etl.
3. Context management: This component is specific to LLMs or multi-modal models. The documents and user requests provide significant value to the answers generated. Ensuring the context management are aligned with the proposed changes to avoid surprising cost or latency increase, or even security breach (a concern that's addressed in QA stage) will be critical in this step as well.

As the focus of the blog is on testing latency and costs, I do not want to get into too much details about each individual components yet. I will post a differerent blog discussion the to-do\'s and to-don'ts in the LLMOps that every Infra team working on LLM-based products should pay attention to. For now, let us move on to discuss the different layers of Latency tests.

## **Latency Tests Strategy**

In general, latency is an important aspect to consider when people consider using an AI-based product. Hence at the dev level, some robust testings are definitely required to prevent latency issues at the start. I have categorized latency tests in several aspects: _parallelism_, _input/output size_, and _service dependency_.

When it comes to parellelism, the scale varies depending on the business, the company size, and the customer requirements. We usually spin up load testing for different parallelism levels in a logarithmic order. For example, a small-sized project, which calls an API endpoint or uses a single inference engine to run the inference, may expect the load testing on a scale of 1/2/4/8/16/32. A project that serves millions of users at the same time, however, may require a significant different scale of testing.

The input/output size layer is more nuanced. Usually we test model's inference speed by running it on specific tasks. However, in my POV, the context of task matters the least when it comes to the input/output size-based latency ablation tests. We can customize a prompt to provide a fixed input size, and at the same expecting a certain length of output by "instructing" the model to do it this way. We may specific a short/medium/long input and pair it with short/medium/long output expectation to derive the latency of the model inference. We need to ensure the data/infra used to run comparisons are completely equal to avoid any impact on token generation so as to isolate this layer perfectly.

Another aspect is whether the service depends on pre/post-processing steps for resources to warmup/cool down/clean up and for input/output to be parsed or buffered. These factors certainly add/reduce latencies in the service during load testing, and we should be able to isolate the impact the proposed changes have on latency from these aspects. Hence, the simulated environment to run the service level load testings should also be identical (the service environment ran today are kept the same from the other day), and A/B testing is encouraged rather than comparing today\'s results with the results genereated 1 month ago.

## **Cost evaluation strategies**

While it may be less of a concern for the Infra team in big tech companies to evaluate the cost-effectiveness of the implemented solution, it is one of the top priorites when it comes to AI startups. When systems and services scale up, the impact of model inferences on the cost grows significantly. Hence cost control is a compulsory task to complete before any QA tests are conducted for the proposed new changes. Usually it comes in three different formats:

- API costs: Sometimes companies rely on third party API providers to run model inferences. However, when the logic of services requires multiple, concurrent model inferences to complete several subtasks, the costs can quickly scale up and becomes uncontrollable. The most fine-grained level of the model cost is based on token usage, as providers often charge based on tokens used. Hence any long input/output pairs need to be validated in the tests.
- GPU resources: Both persistant and on-demand GPU clusters cost a lot for startups to serve their models on. If the GPU utilization is low, or i/o becomes the bottle neck that affects the efficiency of the inference, we are essentially burning money for nothing. Therefore, it is critical to avoid these situations by conducting the right set of simulations and get metrics on costs to validate the changes' impact on cost. No changes should go through to the QA stage if it can potentially blow up the profitability of a product line.
- Electricity and manpower: This category is often ignored, but the manpower required to maintain the proposed changes, and the electricty cost of hosting a large amount of GPU resources for model serving is an implicit killer to some extents.

In light of these aspects, I came up with a strategy to evaluate the cost perspective of the changes:

- Step 1: Test on total token consumptions per request, capture the distribution, identify the anomalies
- Step 2: Test on "useless tokens" generated. For example, if the task output can be easily in bullet point format, using JSON output is not only threatening stability, but also introduces redundant tokens in the inputs and outputs
- Step 3: Test on GPU utilization rate and Memory consumption rate. If "internal fragmentation" happens, we are essentially wasting GPU and thus incurring unnecessary costs.
- Step 4: Analyze the manpower and resources required to adopt the proposed changes and maintain it. Ensure the profit prospects far outweights the cost side of it.
- Step 5: Give a conclusive score on the cost impact of the proposed changes. This step should leave leadership with the right insights on whether to adopt the changes in the final call.

## **Final words**

Once again, the endeavor to further improve the model iteration process does not stop here. As we progress into the final stage in dev, we are going to explore the most widely discussed topic, prompt engineering. This is another effort from the previous prompt engineering whitebook, from a dastically different point of view. It\'s going to be a fun blog to read. Before that blog comes out, **_Stay Hunger, Stay Foolish_**.
