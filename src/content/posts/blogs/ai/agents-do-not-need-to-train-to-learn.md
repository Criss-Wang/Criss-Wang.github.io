---
title: "Agents Do Not Need To Train To Learn"
excerpt: "The model weights can stay frozen while an agent learns through the files, rules, logs, and habits it leaves behind."
date: 2026/06/07
categories:
  - Blogs
tags:
  - AI
  - Agents
  - Memory
  - Systems
layout: post
toc: true
---

Memory is becoming a bad word for what agents do.

It sounds too harmless. A notebook. A few facts saved for later. The user likes short answers. This repo uses `pnpm`. The test command is weird. The deployment script lives in a place nobody would guess.

Those facts matter, but they are not the whole thing.

The more interesting change is that an agent can start a future session with different behavior even when the model weights have not changed at all. The base model is the same. The surrounding world is not.

There are new files. There are command logs. There are project rules. There is a failed approach written down somewhere. There is a hook that blocks a dangerous edit. There is a skill the agent can call instead of improvising. There is a test that did not exist yesterday. There is a scratch note that says, in effect: do not waste another hour here.

That is learning.

It is not training in the usual sense. Nobody updated the weights. But past experience changed future behavior through durable external artifacts. That is enough to make the word "learn" useful, as long as we do not pretend all learning has the same shape.

The practical question is not whether agents can learn without retraining. They already can.

The practical question is what kind of learning we are willing to let them keep.

## The Learning Outside The Model

Most conversations about AI improvement still collapse back to the model. Is the base model smarter? Did the benchmark move? Did the post-training recipe improve? Those questions matter, but they miss the part of the system I keep seeing in real work.

A coding agent is not just a model. It is a model sitting inside a small operating environment: a repo, a shell, a filesystem, tests, instructions, permissions, command history, tool outputs, review comments, and whatever memory system the product gives it.

When that environment changes, the agent changes.

If one session discovers that a migration script silently rewrites generated files, the next session can avoid touching those files directly. If one run adds a failing test for the bug, the next run inherits a sharper definition of the problem. If a user keeps correcting the agent for skipping visual QA, that correction can become a project habit. None of this requires retraining.

The workspace starts acting like an external nervous system.

That is why I do not like treating agent memory as a cute product feature. It is not just a place to store facts. It is one layer in a larger behavioral system.

A transcript says what happened. A memory says what may matter later. A rule says what must happen. A hook enforces something outside the model's discretion. A command packages a workflow so the agent does not reinvent it every time. A skill turns a repeated pattern into a capability.

Those artifacts have different jobs.

"This repo uses `pnpm`" is a memory.

"Run the test suite before claiming the task is done" is closer to a rule.

"Use the security reviewer before touching auth" is orchestration policy.

"When CI fails, run this exact triage sequence" is a command.

Putting all of that under the word "memory" makes the system sound softer than it is. Some of these artifacts do not merely remind the agent. They steer it.

## Memory Is Not Innocent

The danger is that memory feels cheaper than it is.

An agent can finish a task and still write a terrible memory. It can store a temporary workaround as a permanent fact. It can turn one correction into a global preference. It can preserve a mistaken diagnosis because the final answer sounded confident. It can remember too much and make the next session worse.

This is not a theoretical problem. Bad memory has a different failure mode from a bad answer. A bad answer is local. A bad memory becomes future context.

That is why memory quality needs its own standard.

For task completion, we usually ask whether the change worked. Did the tests pass? Did the user get what they asked for? Did the agent avoid making a mess?

For memory, the question is stranger: did the agent save the smallest thing that will improve future behavior, with the right scope and the right amount of uncertainty?

That last part matters. Scope is where memory systems quietly become dangerous. A note that is true in one repository may be wrong in another. A preference that helps one user may annoy another. A debugging pattern that saved one stack may damage a different stack.

The memory system has to know not just what it knows, but where that knowledge is allowed to act.

I like "instinct" as a middle word here. It is not formal, but it points at something useful. An instinct is a memory that has started shaping behavior, without becoming a hard rule. It lives somewhere between fact and enforcement.

That middle layer is powerful. It is also exactly where overreach happens.

## Dreams Are Garbage Collection

The agent should not be doing all of its memory hygiene in the middle of a task.

During a task, the agent is under pressure. It is trying to make progress. It is reading errors, editing files, choosing tools, and deciding what to do next. That is a bad moment to decide what the system should carry forward for weeks.

The language of "dreams" is useful for exactly this reason. Anthropic's managed-agent [Dreams documentation](https://platform.claude.com/docs/en/managed-agents/dreams) describes an offline pass over past sessions and memory, producing a separate output memory store rather than directly mutating the original one. The details may change, but the architectural idea is right: consolidation should be reviewable and discardable.

The dream is where the system can ask slower questions.

Was this correction repeated, or did it happen once? Did a newer session contradict an older memory? Is this really a user preference, or just a fact about one task? Should this stay as memory, become a rule, or disappear?

The best dream is not the one that remembers everything. It is the one that throws away the seductive junk.

I want agents that can say, after looking across ten sessions: this pattern is real, this one was noise, this old belief is stale, and this workflow is stable enough to become a command.

That is very different from "the agent remembers things."

It is closer to maintenance.

## Context Is Not Memory

Longer context windows do not solve this.

A long context window lets the model see more text. That is useful. It is not the same as deciding what deserves to persist. Context is exposure. Memory is selection. Rules are authority. Hooks are enforcement.

If something must happen, memory is too weak.

Use a test. Use a hook. Use a permission boundary. Use a review step. Use a command that makes the desired path easier than improvisation.

Claude Code's [project memory documentation](https://docs.anthropic.com/en/docs/claude-code/memory) makes this distinction visible by separating project instructions, memory, path-scoped rules, and hooks. That is the right design direction. The artifacts should not all have the same force.

The mistake is letting a weak artifact do a strong artifact's job.

If the agent should remember that a repo uses `pnpm`, memory is fine. If the agent must never deploy without running a smoke test, memory is not enough. If the agent keeps doing a workflow badly, the answer may not be another note. It may be a command, a checklist, a test, or a smaller tool surface.

That is the part I think will separate serious agent systems from toy ones. Not the existence of memory. The routing of experience into the right kind of artifact.

## The World Has To Keep Score

Reset-style loops make this obvious.

The Ralph loop is a useful example: run an agent repeatedly against a stable objective, with each iteration starting fresh but inheriting the changed workspace. The prompt may be the same. The world is not. Ralph's own description of [The Loop](https://wiggum.dev/concepts/the-loop/) is basically an argument for externalized state: files, tests, git history, progress markers, and the codebase itself carry the work forward.

This can look almost magical the first time it works. The agent exits. Another run begins. Somehow it continues.

But the trick is mundane. The world is keeping score.

If iteration one writes a test, iteration two sees it. If iteration one records a failed approach, iteration two can avoid it. If iteration one leaves a progress file that says exactly what is still unresolved, iteration two does not need to rediscover the whole situation.

Without that scorekeeping, the loop becomes expensive repetition. The agent starts over, rereads the same files, makes the same plan, hits the same blocker, and produces another polished status update.

That is not autonomy. It is amnesia with a nice final answer.

The agent only looks persistent when the environment is persistent in the right ways.

## Make The Learned Parts Reviewable

Once learned artifacts shape future behavior, they become operationally closer to code than chat history.

They may not compile. They may not have tests. But they affect what the agent will do next week.

That means they need review.

I do not mean every tiny memory needs a meeting. I mean broad or behavior-changing artifacts need a way to be inspected. What changed? Why did it change? Where does it apply? How confident are we? How do we remove it?

A memory diff should be reviewable. A proposed rule should name its scope. A command should show the workflow it automates. A skill should have examples. A subagent should say when it should be invoked and when it should stay out of the way.

Otherwise the system gets a hidden policy layer.

That hidden layer may help for a while. Then it will start surprising people.

The next serious step in agents may not be longer autonomous runs or even larger context windows. It may be better machinery for turning experience into durable behavior without laundering every observation into authority.

Agents do not need to train to learn.

But if they are going to learn through memory, dreams, rules, commands, skills, and the workspace itself, we need to decide what they are allowed to keep.
