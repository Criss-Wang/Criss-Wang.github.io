---
title: "A Brief Intro to A/B Testing"
date: 2019/12/20
categories:
  - Blogs
tags:
  - Statistics
  - Experimentation
excerpt: "A practical introduction to A/B testing: hypotheses, metrics, randomization, sample size, statistical significance, and common experiment pitfalls."
layout: post
mathjax: true
toc: true
---

## Introduction

A/B testing compares two or more product variants by randomly assigning users or events to each variant and measuring whether the change improves a predefined metric.

The core idea is simple:

- Group A sees the current experience.
- Group B sees the new experience.
- Random assignment makes the groups comparable.
- A metric decides whether the change helped.

The hard part is not the math. The hard part is choosing the right metric, avoiding biased assignment, and interpreting results without fooling yourself.

## Define the Hypothesis

An experiment should start with a specific hypothesis:

```text
Changing [feature] from [old behavior] to [new behavior] will improve [primary metric] for [population] because [reason].
```

Example:

```text
Changing the signup button copy from "Submit" to "Create account" will improve completed signups for new visitors because the action is clearer.
```

If the hypothesis is vague, the result will be vague too.

## Choose Metrics

Pick one primary metric before the test starts. This is the metric used for the launch decision.

Common primary metrics:

- Conversion rate.
- Click-through rate.
- Revenue per user.
- Retention.
- Task completion.
- Latency.
- Error rate.

Also define guardrail metrics. A change might increase clicks while hurting retention, trust, or system performance.

Good guardrails include:

- Page latency.
- Error rate.
- Refunds.
- Complaints.
- Unsubscribe rate.
- Downstream conversion.

## Randomization

Randomization is what makes the comparison credible.

Decide the unit of randomization:

- User-level randomization for user experience changes.
- Session-level randomization for short-lived experiences.
- Request-level randomization only when cross-request interference is not a concern.
- Cluster-level randomization when users influence each other.

Do not mix units casually. If the same user can see both A and B, the test may measure confusion instead of product quality.

## Statistical Test

For a conversion metric, the basic comparison is the difference in proportions:

$$
\Delta = p_B - p_A
$$

where $p_A$ is the conversion rate in control and $p_B$ is the conversion rate in treatment.

The usual question is:

```text
Is the observed difference large enough that random noise is an unlikely explanation?
```

Statistical significance is useful, but it is not the whole decision. Practical significance matters too. A tiny lift can be statistically significant and still not worth launching.

## Sample Size and Power

Small tests are noisy. Before running the experiment, estimate:

- Baseline conversion rate.
- Minimum detectable effect.
- Significance level.
- Desired power.
- Expected traffic.
- Experiment duration.

If the test is underpowered, it may fail to detect a real improvement. If it runs too long, external seasonality and product changes can contaminate the result.

## Common Pitfalls

### Peeking Too Often

If you repeatedly check significance and stop the moment the result looks good, the false-positive rate increases. Use a planned duration or sequential testing method.

### Too Many Metrics

If you test many metrics and only report the one that moved, you are likely overfitting the analysis. Decide the primary metric ahead of time.

### Sample Ratio Mismatch

If the expected split is 50/50 but traffic lands 60/40, something may be wrong with assignment, logging, or filtering.

### Novelty Effects

Users may react differently simply because something is new. For retention or long-term behavior, short tests can mislead.

### Interference

One user's treatment can affect another user's outcome. Social feeds, marketplaces, ads, and recommendation systems are especially vulnerable.

## Launch Decision

A good launch review asks:

- Did the primary metric improve?
- Did guardrails stay healthy?
- Is the effect practically meaningful?
- Was the sample ratio correct?
- Were there logging or instrumentation issues?
- Is the result consistent across important slices?
- Are there product reasons to delay or roll out gradually?

The best A/B tests do not only answer "which button wins?" They build a habit of making product decisions with evidence.
