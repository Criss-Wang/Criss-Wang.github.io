---
title: "Fine-Tuning in LLMs"
date: 2024-02-19
categories:
  - Blogs
  - Draft Notes
tags:
  - LLM
  - Fine-tuning
excerpt: "A practical overview of supervised fine-tuning, LoRA, prompt tuning, adapters, RLHF, DPO, data quality, and evaluation for large language models."
layout: post
toc: true
---

## Introduction

Fine-tuning adapts a pretrained language model to a more specific task, behavior, style, or domain. It is not always necessary. Many problems are better solved with prompting, retrieval, tools, or product design.

Fine-tune when you need:

- Consistent output format.
- Domain-specific style.
- Task behavior that examples cannot reliably induce.
- Lower latency or smaller prompts.
- A model that performs well on repeated, narrow workflows.

Do not fine-tune to teach the model large amounts of changing factual knowledge. Retrieval is usually a better fit for that.

## Main Approaches

### Supervised Fine-Tuning

Supervised fine-tuning (SFT) trains the model on input-output examples.

Use it for:

- Classification phrased as generation.
- Structured extraction.
- Customer support response style.
- Domain-specific summarization.
- Tool-call formatting.
- Instruction following in a narrow domain.

The dataset matters more than the training trick. A few thousand clean examples can beat a much larger noisy dataset.

### LoRA and QLoRA

LoRA trains small low-rank adapter matrices while keeping most base-model weights frozen. It is popular because it reduces memory and makes multiple task adapters easier to manage.

QLoRA goes further by loading the base model in low-bit precision, commonly 4-bit, while training adapters. This can make fine-tuning feasible on much smaller hardware.

Use LoRA or QLoRA when:

- You want cheaper fine-tuning.
- You need multiple task-specific variants.
- You do not want to store a full copy of every model.
- You can accept adapter-based deployment complexity.

### Prompt Tuning and Prefix Tuning

Prompt tuning trains a small set of continuous prompt embeddings. Prefix tuning trains learned prefix states that steer the model.

These methods are parameter-efficient, but they can be less intuitive to debug than LoRA or supervised fine-tuning. They are useful when the base model should remain frozen and the task is narrow.

### Adapters

Adapters insert small trainable modules inside the model. Like LoRA, they keep most pretrained weights frozen.

Adapters are useful when:

- Multiple tasks share one base model.
- You want modular task-specific behavior.
- Full fine-tuning is too expensive.

## Preference Optimization

Instruction models often need preference data, not just correct answers.

### RLHF

Reinforcement learning from human feedback (RLHF) typically trains a reward model from preference comparisons, then optimizes the policy model against that reward model.

It can improve helpfulness and alignment, but it is operationally complex:

- Human preference collection is expensive.
- Reward models can be gamed.
- Training can be unstable.
- Evaluation must include safety and quality review.

### DPO

Direct Preference Optimization (DPO) optimizes directly from preference pairs without training a separate reward model in the classic RLHF style.

It is often simpler to run than RLHF and is widely used for aligning model behavior with preference data. The dataset still matters: bad preference pairs create bad behavior.

## Data Quality

Fine-tuning data should be:

- Task-specific.
- Cleanly formatted.
- Deduplicated.
- Representative of production inputs.
- Reviewed for unsafe or private content.
- Split into train, validation, and test sets.
- Evaluated by slices, not only aggregate score.

Common data mistakes:

- Training on outputs that are too verbose or inconsistent.
- Mixing incompatible instruction styles.
- Letting evaluation examples leak into training.
- Fine-tuning on stale facts.
- Forgetting refusals, edge cases, and malformed inputs.

## Evaluation

Evaluate the model before and after fine-tuning:

- Task success rate.
- Format validity.
- Factuality or source faithfulness.
- Safety behavior.
- Latency and cost.
- Regression on general capabilities.
- Performance on hard slices.
- Human preference review for subjective tasks.

For structured outputs, add strict parsers and schema tests. For retrieval-augmented systems, evaluate retrieval and generation separately.

## Deployment Concerns

Fine-tuning changes the operations story:

- Which base model and adapter version are active?
- How are adapters stored and promoted?
- Can the system roll back quickly?
- Does inference support adapter loading efficiently?
- Are prompts still needed?
- How are fine-tuning datasets tracked?
- How will new preference data be collected?

The model registry should record base model, adapter, training data, training config, and evaluation report.

## A Practical Decision Path

Use this order:

1. Improve prompting.
2. Add retrieval if knowledge is missing or changing.
3. Add tools if the model needs external actions.
4. Fine-tune with SFT if behavior is repeated and examples are available.
5. Use LoRA or QLoRA if full fine-tuning is too expensive.
6. Add preference optimization if output quality depends on subjective preferences.

Fine-tuning is powerful, but it is not a substitute for product clarity, clean data, and good evaluation.

## References

- [Hugging Face PEFT](https://huggingface.co/docs/peft/index)
- [Hugging Face TRL](https://huggingface.co/docs/trl/index)
- [LoRA](https://arxiv.org/abs/2106.09685)
- [QLoRA](https://arxiv.org/abs/2305.14314)
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
