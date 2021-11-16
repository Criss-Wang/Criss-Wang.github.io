---
title: "A brief Intro to A/B Testing"
date: 2019-12-20
layout: single
author_profile: true
categories:
  - Statistics
  - Business
tags: 
  - A/B testing
  - Hypothesis Testing
  - Product Pipeline
excerpt: "A Recap of Fundamental Concept: A/B Testing "
mathjax: "true"
---
## Introduction
A/B testing is a type of hypothesis testing commonly used in business. It is mainly useful on for products that are mature and is suitable for fast iterative product development. The main idea is to assume a new modification is useful and take trials/experiments upon the modified product. Next, using the old version as a reference, see how significant is the improvement brought by the new modification. 

**Note:** When we say *new modification*, it must be warned that the change should contain only **one** factor, otherwise the influence can be compounded.
{: .notice--info}
## 10 Steps in A/B Testing
1. First time trying something new: run an A/A testing simultaneously to check for `systematic biases`
  -  `Systematic Bias`: is sampling error that stems from the way in which the research is conducted. There are three types: <br/>
    a.  Selection bias: Biased way to select the factors/samples. <br/>
    b.  Non-response bias: the participants/customers involved in your tests are different in their behavioral patterns from the people that are not involved in your study (other general public) <br/>
    c.  Response bias: The result/inference that are given does not follow the truth/real observations.
2. Define the goal and form __hypothesis__
  - `Null hypothesis` $H_0$: a general statement or default position that there is __no relationship__ between __two measured phenomena__. Often claimed to be the case when there is no association between the performance of product and the a feature change of the product. If we believe otherwise, then we are arguing for a `Alternative hypothesis`  $H_a$.
3. Identify __control__ and __treatment__ groups
4. Identify __KPI/metrics__ to measure
  - e.g: *click through rate*, *conversion rate*, *renewal rate*, *bounce rate*, *average retention*, etc...
5. Identify what data needs to be collected
  - Sometime people use the __Customer Funnel analysis__:
    <figure style="width: 300px" class="align-center">
    <img src="/images/Data Science Concept/funnel.png" alt="">
    <figcaption>credit: https://clevertap.com/blog/funnel-analysis.</figcaption>
    </figure> 
  - Example: [__Netflix - Funnel Description__] Customer will enter the home page, which invites customer to further enter via the button `free trial for one month`, then customer will try and then register and pay 
  - Example: [__Netflix - Funnel Analysis__]Funnel analysis: here the funnel will be how the product will affect each steps like `improving converted customer size` or `improving returning customer size`, during the procedure to reach __click through/subscription__, what side-effects/main effects are triggered
  <br>
6. Make sure that appropriate __logging__ is in place to collect all necessary data
7. Determine how small of a difference can be (__define significance level__ and thus __power__ of the experiment)
  - `Significance Level`: or *p*-value, is the probability that we reject the null hypothesis while it is true (type I error). 
  - `Power`: is the probability of rejecting the null hypothesis while it is false (type II error).
8. Determine what fraction of visitors should be in the treatment group (__control/treatment split ratio__)
9.  Run a [power analysis](https://stats.idre.ucla.edu/other/mult-pkg/seminars/intro-power/) to decie how much data is needed to collet and how long to run the test
  - compute __running time__ given customer flow
  - Run the test for __AT LEAST__ this __running time__ long

## Before you run A/B testing...
### Why should you run a A/B test?
- A/B testing is the key to understand what drives your business and make data-informed business decisions
- To understand the __causal relationship__ and not simply the correlations

### When to run experiments
- Deciding whether or not to launch a new product or feature
- To quantify the impact of a feature or product
- Compare data with intuition (Understand how users respond to certain parts of a product)

### When not to run experiments
- No clear comparison between the control and experimental group
- Emotional changes need time: Logo/Brand name
- Response data hard to obtain
- Too time consuming/costly

## A closer look at Type I and Type II errors
### Terminology 
- `significance level`: The significance level for a given hypothesis test is __a value for which a *p*-value less than or equal to__ is considered __statistically significant__. Often denoted by __α__
- `region of acceptance`: The __range of values__ that leads the researcher to __accept the null hypothesis__. [`significance level`↓ $\implies$ `region of acceptance`↑ : harsher condition to produce a winner]
- `effective size`: The __difference between the true value and the value specified in the null hypothesis__. [Effect size = True value - Hypothesized value]

### Type 1 Error
- When the `Null Hypothesis` is true but rejected
- 'false positive' - happen when the tester validates a __statistically significant difference__  even though there isn’t one __(no winner situation)__. `positive` here means a __valid winner__ concluded
- Type 1 errors have a probability of __"α"__ correlated to the level of confidence that you set. A test with a 95% confidence level (__α = 0.05__) means that there is a __5% chance of getting a type 1 error__. 
- __[In business sense]__ This means that you will wrongfully assume that your hypothesis testing has worked even though it hasn’t.
- __[The business consequence]__ Potential loss of money after adjustment is made, because __your variation didn’t actually beat your control version in the long run__.

### Type 2 Error
- When the `Null Hypothesis` is false but accepted
- 'false negative' - Happens when you inaccurately assume that no winner has been declared between a control version and a variation __although there actually is a winner (can be either)__
- The probability of a type 2 error is __"β"__. __β__ depends on the __power__ of the test (i.e the probability of not committing a type 2 error, which is equal to __1-β__).
- There are 3 parameters that can affect the power of a test:
    - `Sample size (n)`: Other things being equal, the _greater the sample size_, _the greater the power of the test_ (but also _the more expensive of the test_).
    - `Significance level of test (α)`: The lower the `significance level`, the lower the power of the test.
        - If you reduce the `significance level` (e.g., from 0.05 to 0.01), the `region of acceptance` gets bigger. As a result, you are less likely to reject the null hypothesis. This means you are less likely to reject the null hypothesis when it is false, so you are more likely to make a Type II error. In short, the power of the test is reduced when you reduce the significance level; and vice versa.
    - The "true" value of your tested parameter or `effective size`:  The greater the difference between the "true" value of a parameter and the value specified in the null hypothesis, the greater the power of the test.
- __[The business consequence]__ Potential loss of money after adjustment is made, because __your variation didn’t actually beat your control version in the long run__

## Toolkits
1. Usually the test plan/create variation step can be executed with company's own techpack, or using some popular tools
- __Google optimize__
- __optimizely__
2. For hypo testing and result analysis: online resources of excel's A/B testing macro are widely available