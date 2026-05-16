---
date: 2024-10-24
layout: post
title: "ResumeAssist"
categories:
  - Projects
  - Python
  - Next.js
  - FastAPI
  - LLM Agents
excerpt: "A private, configurable AI resume assistant with LangChain agents, CV rendering, and a full-stack review workflow"
link: "/images/Projects/resumeassist-diagram.svg"
mathjax: true
toc: true
---

### Introduction

[ResumeAssist](https://github.com/Criss-Wang/ResumeAssist) is a personal resume assistant built around configurable AI suggestions rather than a single fixed prompt. The current version combines a Next.js interface, FastAPI backend, LangChain agent logic, Neo4j storage, and CV rendering into a private workflow for iterating on resumes.

### What is in the repo

- Anthropic-backed LangChain engine
- Config-based agent system
- Basic resume modification UI
- Backend and frontend integration
- Neo4j connection for structured resume context
- Unit testing, linting, and coverage workflow

### Where it is going

The roadmap points toward local model support, GraphRAG, observability with evaluation tooling, and a more polished UI. The interesting bit is the shape: a resume tool that keeps domain context structured enough for agentic editing, not just a chat box around a PDF.
