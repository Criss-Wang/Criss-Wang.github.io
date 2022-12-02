---
date: 2022-07-31
layout: post
title: "Code Analyzer"
categories:
  - Projects
excerpt: "C++ based code static program analyzer"
mathjax: true
toc: true
---

### **Introduction**
A Static Program Analyzer (SPA) is an interactive tool that automatically answers queries about programs. It is my [CS3203 Software Engineering Project](https://docs.google.com/document/d/1sIwr_8Li6660Snw5F9VHbGkIO1CYl0lE/edit?usp=sharing&ouid=101107396415895765675&rtpof=true&sd=true) product. Our team designed and implemented a SPA for a simplified programming language (SIMPLE) with specific query language (PQL). Here is a glimpse of [our work](https://github.com/Criss-Wang/Code-Analyzer).

<figure align="center">
    <img src="/../../images/Projects/PKB.png" width="500px">
</figure>

- Major contributions: 
    - **System design & API specification**
    - Designed customized **Postfix String Conversion** algorithm for design extractor & population
    - **Program Knowledge Base (PKB)** development
    - Fast lexical token querying via **Indexing and Concurrency (C++ STL threads)**
    
- DevOp related:
    - Extensive unit testing/integration testing
    - CI/CD management with CMake and Github workflow
    - Documentation

### **Tech & Methodology**
<div>
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/cplusplus/cplusplus-original.svg"  width="40" height="40"/>&nbsp;
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/cmake/cmake-original.svg" width="40" height="40"/>&nbsp;
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/bash/bash-plain.svg"  width="40" height="40"/>&nbsp;
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linux/linux-original.svg" width="40" height="40"/>&nbsp;
</div>

- C++ STL threads
- Unit Testing/Integration Testing with [Catch](https://github.com/catchorg/Catch2)
- System Testing
- Github workflow