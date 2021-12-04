---
title: "NLP: Intro"
date: 2020-07-05
layout: single
categories:
  - Natural Language Processing
tags: 
  - Machine Learning
  - NLP
  - Deep Learning
excerpt: "Brief intro to natural language processing"
mathjax: "true"
---

## Overview
Natural Language Processing (NLP) is a field in machine learning that aims to solve the following problems:
- **N**atural **L**anguage **U**nderstanding
- **N**atural **L**angauge **G**eneration


![NLP-problems]({{ site.url }}{{ site.baseurl }}/images/Machine learning/NLP2.png)

## 1. Main Obstacles
- __Sparcity__: The main difficulty in generating an NLP model is to generate a dense-enough matrix to include all the words. Due to the large vocabulary of words, we cannot simply do _[one-hot-encoding ](https://en.wikipedia.org/wiki/One-hot "One-hot wiki")_ on it, as it will cause the matrix to be very sparse and inefficient.
- __Ambiguity__: A sentence may be interpreted differently based on its context. A word may have multiple meanings/usages. A phrase may need to be interpreted separated from other single words. We also have evolving langauges like LOL, Googling, emoji etc. Handling these ambiguous cases in a sentence is a challenging task. Below are 2 simple examples:
    - Example 1: 'I saw her duck'
        1. I | saw (see) | her duck
        2. I | saw (cut) | her duck
        3. I | saw (see) | her duck (squad)
    - Example 2: 'The Pope's baby steps on gays'
        1. The Pope's | baby steps | on gays
        2. The Pope's baby | steps on gays

## 2. Some Popular mature toolkits
- Word Embedding (Use numerical vectors to preserve key information of a word)
    - Word2Vec ([Play with its Visualization](https://projector.tensorflow.org/ 'Embedding projector'))
    - GloVe
    - FastText
- Transformer
    - TO be udpated...

## 3. NLP pipeline
The pipeline of NLP has evolved over the years, the traditional way of processing the langauge model is like this:
<figure style="width: 600px" class="align-right">
  <img src="{{ site.url }}{{ site.baseurl }}/images/Machine learning/NLP_pipeline.png" alt="">
  <figcaption>Traditional NLP pipeline.</figcaption>
</figure>

1. We have either __speech/voice__ or __text__ input, beeing processed to become raw data.
2. Do a senquence of analysis (Morphological and lexical analysis, followed by syntactic analysis) on the sentence in raw data to generate a tree of words/phrases
3. Conduct other types of study (semantic interpretation, sentimental analysis) from the leaf nodes of the tree

Currently, most SOTA project of NLP are using the following pipeline:
<figure style="width: 610px" class="align-right">
  <img src="{{ site.url }}{{ site.baseurl }}/images/Machine learning/NLP_pipeline_new.png" alt="">
  <figcaption>New NLP pipeline.</figcaption>
</figure>

Here I will not explain the details as they quite self-explanatory if you know a little bit about machine learning common pipeline. 

## 4. Applications
Common usages of NLP:
- Machine Translation:
    - language translate
    - Outcome evaluation (BLEU): Low = Bad, High != Always good
- Information Retrieval (Search)
    - User Customized
    - Common sense: find key-word assoicated result and directly output them together with the searched link
    - Question Answering: A very recent discovery
- Search AutoComplete
    - Aggregated search history
    - Customized
    - Nullstate
- Text Categorization
    - Spam Filter
    - Sentimental Analysis (Upstream Signals)
    - Dialogue Agent/Chat bot
- Entity Detection
    - Entity Linking (identity what each word/phrase represent)
        - Apple: Organization
        - Steve Jobs: Person
        - April: Date
- Spelling and Grammer Checker
- Interesting services (Work report)
    - IBM Watson Tone Analyzer
    - Amazon Comprehend
    - Microsoft LUIS