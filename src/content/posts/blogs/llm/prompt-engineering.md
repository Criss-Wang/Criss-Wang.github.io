---
title: "Prompt Engineering Whitebook"
excerpt: "A copy of what I learned and gathered about prompting, both from online and from work"
date: 2024/3/20
categories:
  - Blogs
tags:
  - LLM
layout: post
mathjax: true
toc: true
---

## Why prompting

When working with LLMs, the rule number one is: **Don\'t touch the model**. Very often, people (especially students with more experience in model tuning and less industrial-level prompt engineering experiences) will opt for finetuning when they have a new problem at hand. However, the harsh reality is that most real-world problems are either simple enough to handle with a good prompt, or complex enough that fine-tuning on available large datasets become less effective.

In my opinion, prompting should ideally be your first approach. Complex tasks can often be decomposed into smaller, easier tasks and solved with pretrained models. Yuo should only go changing model architecture once your prompts are as good as they can be. No company would want to burn money at start, only to realize that easy solution with prompt engineering is there lying on the table.

Major benefits of prompt engineering include:

1.  Reduce costs by moving to a smaller model
2.  Eliminate finetuning costs
3.  Enable lower-latency communication by changing the general format

## Assumptions

This blog assumes basic understanding of prompting, such as what forms a prompt, what are different components of a prompt, and how prompts are transformed into tokens for model inferences. You may check online resources for it if you don\'t now about them.

## Techniques

### Use Templates

Most open source model have their specific prompt tempaltes. You can refer to their website, or find it on Hugging Face. Some basic ones include

- [Vicuna-13B](https://huggingface.co/CarperAI/stable-vicuna-13b-delta)

```js
### Human: your prompt here
### Assistant:

```

- [Llama-2-chat](https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML)

```js
[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don\'t know the answer to a question, please don\'t share false information.
<</SYS>>
{prompt}[/INST]

```

- [EverythingLM-13B-16K](https://huggingface.co/TheBloke/EverythingLM-13b-V2-16K-GGML)

```js
You are a helpful AI assistant.

USER: {prompt}
ASSISTANT:

```

- [Claude](https://claude.ai/)

```js
Human: Human things
Assistant: {{Response}}
```

Things to take note of:

- Some models don\'t have system prompts
- Some models will prefer alternating between `user` prompt and `assistant` prompts
- When choosing models, you should often look out for instruction-finetuned models, as they are the most prevalent ones for chat completion/streaming chat tasks.

### Few-shot learning

To put it in a user-friendly text, _few-shot learning_ in the context of llm prompting is simply providing example into the prompt. You may raise a question or instruction for the llm/chatbot to answer. But sometimes they don\'t know the answer format or they hallucinate without suffcient context. Giving some examples often helps models to understand the instructions better, thus providng a more cohesive and relevant answer. As an example, suppose one wants LLM to output a JSON, but on the first try, the JSON was malformed. To fix this issue, one can either pass the output to LLM and ask it to fix it by itself, or he can retry using a better prompt with few-shot learning:

```js
### Human: Given a sentence "This place is horrible" from Wall Street Journel, determine if it has positive/negative/neutral sentiment. Output the result in JSON format. Here are a few examples:
{"sentence": "The food is enjoyable", "sentiment": "positive"}
{"sentence": "Princess Kate was diagnosed with cancer", "sentiment": "netural"}
{"sentence": "War criminials need to be punished heavily", "sentiment": "negative"}
### Assistant:
```

If we want the model to learn to merge queries from past responses and better the answer, we can improve the prompt above by splitting in into conversations

```js
### System: Given a sentence from Wall Street Journel, determine if it has positive/negative/neutral sentiment. Output the result in JSON format.
### Human: The sentence is "The food is enjoyable", the output JSON is:
### Assistant: {"sentence": "The food is enjoyable", "sentiment": "positive"}
### Human: The sentence is "Princess Kate was diagnosed with cancer", the output JSON is:
### Assistant: {"sentence": "Princess Kate was diagnosed with cancer", "sentiment": "netural"}
### Human: The sentence is "War criminials need to be punished heavily", the output JSON is:
### Assistant: {"sentence": "War criminials need to be punished heavily", "sentiment": "negative"}
### Human: The sentence is "The place is horrible", the output JSON is:
### Assistant:
```

You can also teach multi-turn behavior - like adding together queries, and cleaning them out when requested via this few-shot learning technique.

With all these benefits, we must not ignore its potential **problems**:

- Model often struggles to move away from pre-training knowledge
- It significantly uses up the token budget of your prompts, which can be expensive
- Sometimes giving examples is counter-effective. For example, providing a single positive example can cause the model to always output positive label. Providing two positive and one negative can cause the model to think the next one must be negative. Sometimes this pattern happen because the label distribution is very skewed. Sometimes it could be domain knowledge issue as well. Be sure to check it out and eliminate potential hallucination issues when applying this technique.

### Manage prompt complexity

Suppose you are talking to a human and providing instruction to them. If you provide a long, complex set of instructions in one shot and expects the human to follow it, how confident are you in him/her completing the instruciton as you wanted? Most cases it achieves nothing but anger in that person\'s mind. Now think about the case when you talk to a chatbot, the sheer complexity of prompt can also be countereffective from time to time. Hence, managing the complexity of your prompt is a really important part of prompt engineering. Here are a list of things I recommend checking to achieve a good balance when managing your prompts for your tasks. Most prompts have three primary types of complexity and we will handle them one by one.

**Task Complexity**

- Definition: Difficulty of the major task
- Example: `Who are the characters in this text` is significantly simpler than `Identify the key antagonists`
- How to reduce it:
  - Break it down to smaller, simpler tasks
  - Insert a chain of thought before asking for an answer. `Think step-by-step` is an easy addition
  - Pointing out which part of the problem to solve first. Models need to know where to start and start the right way.
  - Sometimes you can debug model\'s thought process by asking it to print it out

**Inference Complexity**

- Definition: The amount of inference the model needs to _understand_ your task.
- Counterintuitively, this is something that affects small, seemingly simple prompts.
- Example: understanding what is an _intent_ can be tough, as it can mean general objectives in research, or enquiry in customer service.
- How to reduce it:
  - Provide explanation/definition to those keywords
  - Switch to a simpler/general words if possible
  - Often requires prompt size to grow
  - Ask the model to define it himeselve to achieve implicit chain-of-thought

**Ancillary Functions**

- Definition: smaller tasks you are explicitly (or implicitly) asking the model to perform
- Examples: transformations to the JSON; retrieving and merging things from previous messages.
- How to reduce it:
  - Prompt Switching: essentially keeping the context and vary the instructions
    - Note: Conversationally tuned models (like llama-2) will prefer this, but other instruction-following models might find it hard to retrieve intermittent context (hiding in between human instructions) when it comes to answering the final, big question.
  - Self-consistency: You can test if the complexity is removed by turn the temperature up if your task permits it, and see if the results are aligned
  - If your prompt works well across multiple models, it\'s a good sign that it\'s well-spelled out

**A checklist for reducing prompt comlexity**

1. Primary task
2. The most valuable thing I need the model to do
3. Key terms in the task: are they very, very well defined, or so simple that there\'s no ambiguity?
4. Any explicit/implicit additional tasks aside from primary task: are they integral to the performance of my primary task? Can I split them into other prompts or find ways to reduce their complexity?
5. Any domain knowledge or things that require domain expertise: can model infer or learn these eccentricities about this domain?
6. Any instruction requirements: is my task a question? does it need instructions (like this list you\'re reading) on how to start towards a solution?

### Spoon-Feeding

Intuition: LLMs are next-token probability predictors, and the sooner you can get them going in the right direction, the more likely that they\'ll follow it.

Example:

```js
Human: Please help this user with his questions, by providing a list of ingredients for his recipe.

Human: I'm making a mud pie!

Assistant: Cool! The ingredients you'll need are
```

Notice in `Assistant` , the tokens all the way up to `are` are fixed, and the next token is our required word.

Note that OpenAI GPTs don\'t support this strategy (but you can still leave uncompleted text at the end for a workaround), but almost every other model and provider does.

### Proper usage of System prompts

Attention to system prompts have always been a potential weakness of GPT models (but may be fixed in later versions). However, Llama-2 class of models actually handle system prompts well, as they use special mechanisms in training (like [Ghost Attention](https://arxiv.org/abs/2307.09288)) to increase the effectiveness of a system prompt to influence a conversation, even after many messages.

Some useful things you can use your system prompts for:

1.  Hold Facts, Rules (see below) or other general purpose information that don\'t change as the conversation proceeds.
2.  Set the personality of the assistant. A strong personality (e.g. `You are a chess grandmaster`) may lead to better quality of the task completed in some cases.
3.  Set (or reinforce) an output format (.e.g `You can only output SQL.`)
4.  Move repeated bits of user messages out so you can do better few-shot learning.
5.  Make changing the task for this prompt easier without editing the conversation history.

### Meaningfully distinct keywords

For some keywords that you want the model to put close attention to, convert the normal natural language to a special format. It is recommended to use `CAPITAL_UNDERSCORED_HEADINGS`. As an example:

```js
The travel document I want you to read:
Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.

Use the travel document provided to extract the key destinations the user is travelling to.

```

Can be transformed into:

```js
USER_TRAVEL_DOCUMENT:
"""
Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.
"""

Extract the key destinations from USER_TRAVEL_DOCUMENT.
```

### Proper escaping

In most cases, the information provided (documents, emails, etc) will be in the same language and follow similar formats to your instructions.

- Use escaped (and Meaningfully distinct keywords) to help the model separate which is which.
- Use backticks (`) or triple quotes (”””) to escape your data sections.
- Use a few recommended formatting options for input/output
  - **Multi-line strings**: pretty easy, use this for unstructured data.
  - **Bulleted lists**: easy way to mark something as a list. Save your tokens unless your experience differs.
  - **Markdown tables**: pretty token heavy. Use these if your data comes in markdown tables, or you need it for easy output formatting.
  - **Typescript**: The significantly better choice for expressing a typespec, especially with comments mixed in.
  - **JSON**: Uses more token than many of the above. But may become the new standard in the long term (OpenAI funciton has JSON formatted output support).
  - **YAML**: Close to natural language. Also pretty conservative on tokens. Not having random curly braces helps the BPE better break up your characters into larger token chunks.
- A _rule of thumb_: If you want support, use JSON. If you want brevity, use YAML. If you have a typespec, use Typescript. If it\'s simple, just separate with newlines.

### Content structuring with Facts and Rules

Sometimes structuring your prompt may make your prompts easier to read for both you and the model. Aside from proper escaping, we often use facts and rules to guide models to complete the task:

- **Facts** list what the model should presume before working on the task. Organizing your prompts this way helps you better understand and modify them later on (and prevent prompt-rot)
- **Rules** are specific instruction to follow when executing on a task
  An example can be:

```js
FACTS:
1. Today's date is 6 September 2023.
2. Pax, pp, per person all mean the same thing.

RULES:
1. You need to outline your logical premise in Prolog before each sentence.
2. Write the text as the user, not on behalf of them.
```

### Chain-of-Thought

This is a well-known method, I\'ll just pass two examples with different tasks for inspiration

1. Cliff-summarising a story

Let\'s say you want to take a story and summarise the key story beats. You keep trying but the LLM keeps missing things. Here's one approach.

```js
STORY:
"""
Just wakin' up in the mornin', gotta thank God
I don't know but today seems kinda odd
No barkin' from the dog, no smog
And momma cooked a breakfast with no hog
"""

Summarise this story into the key plot points.

```

One way to improve effectiveness is to work out how you would do it.

```js
Summarise this STORY into key plot points.

STORY:
"""
Just wakin' up in the mornin', gotta thank God
I don't know but today seems kinda odd
No barkin' from the dog, no smog
And momma cooked a breakfast with no hog
"""

Go step by step to get the plot points:
1. Outline the key players in the story. Who are the characters?
2. List the major plot points and who was involved.
3. For each plot point, list the consequences of this happening.
4. For each consequence, see if there are any story beats missing from the first list, and list them.
5. Resummarise the story in terms of beats, labelling each point as positive or negative and it's contribution to the story.

```

This kind of prompting also produces responses that are far easier to debug.

2. Continuing a story

Now say we wanted to write the next chapter for the same story - a far more creative endeavor. Here\'s a naive prompt:

```js
STORY:
"""
Just wakin' up in the mornin', gotta thank God
I don't know but today seems kinda odd
No barkin' from the dog, no smog
And momma cooked a breakfast with no hog
"""

Write the next chapter of the STORY.

```

Here\'s a better one.

```js
STORY:
"""
Just wakin' up in the mornin', gotta thank God
I don't know but today seems kinda odd
No barkin' from the dog, no smog
And momma cooked a breakfast with no hog
"""

We need to write the next chapter of STORY, but let's go through the steps:
1. List the main characters in the STORY, and what their personalities are.
2. What are their arcs so far? Label each one on a scale of 1-10 for how interesting it is, and how important it is to the main story.
3. List which arcs are unfinished.
4. List 5 new characters that could be introduced in the next chapter.
5. List 5 potential, fantastical things that could happen - major story beats - in the next chapter.
6. Grade the new characters and the new occurrences 1-10 on how fun they would be, and how much they fit within the theme of the existing story.
7. Write the next chapter.
```

### Chain-of-Thought but multi-path automation + validation

When designing chain-of-thought prompt, or any set of facts + rules to better structure your prompt content, consider consulting GPT-4 or other expensive models to get suggestions. The pseudocode is

```
For each COT path (rules/facts):
	Build prompt with these context
	Run inference to get results
	Perform debugging step and generate a score
Select the candidate with the highest score
```

### Some other tricks (To be expanded)

- Pretended that some of our provided context came from the AI and not us. Language models will critique their own outputs much more readily than your inputs
- For each model, use delimiters and keywords that look and feel similar to the original template/dataset used for the model, even if they\'re not directly part of the dataset
- In some cases, asking the model to annotate its own responses with a probability of acceptance, and thresholding this value to remove the worst candidates can improve results.
- Using structured text like pseudocode may improve results
- Replace negation statements with assertions (e.g., instead of "don\'t be stereotyped," say, "please ensure your answer does not rely on stereotypes")
- If budget allows, find a way to express the output in structured format where it can be auto-verified (in polynomial time ideally). Then turn the temperature up and take a few passes through the same prompt. Pick the majority winner.

## How to debug your prompt

- Never pass user input (more specificly, raw customer input) directly to model for output
- Never invent custom formats. Use and modify what\'s already in the lexicon of the model.
- Remove syntax and semantic errors. Sometimes this cause models to output wrong things. Example: saying `output characters` in an instruction may direct model to prefer outputing multiple characters when there should be only one valid character.
- When dealing with specific output format, don\'t put trailing fullstop/coma/semicolon as they may break the output structure.
- Vary the order of your instructions and data to make a prompt work
- Vary where the information is placed (user prompt vs system prompt vs assistant prompt)
- Change the wording, sometimes the keywords/phrases that are domain-specific or abstract are understood by different models differently. check if changing some keywords to its variants or make them clearer can be helpful- When performances of output among different models using the same prompt are similar (sometimes can be done using an LLM evaluator), and you are happy with the results, your prompt is probably ready to use.

## When to modify the model itself

1.  You\'ve tried extensive prompt optimization, and you\'re nowhere near your required success rate.
2.  You need to move to a smaller model, for privacy or cost reasons.
3.  You have a large enough dataset, and the time and money to finetune a model.
4.  Your problem space sits far outside the pretraining dataset - maybe you work in Swift, or you need to train a DSL.
5.  You have a particular style of interaction that you need to "bake in", even at the cost of potentially overfitting.
6.  You need to reverse some prior finetuned behavior.

## References

- [Everything I know about Prompting](https://olickel.com/everything-i-know-about-prompting-llms#properescaping)
