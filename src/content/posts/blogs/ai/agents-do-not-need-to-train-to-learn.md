---
title: "Agents Do Not Need To Train To Learn"
excerpt: "The model weights can stay frozen while an agent learns through memory, rules, commands, skills, logs, and the workspace around it."
date: 2026/06/07
updated: 2026/06/08
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

There are new files, command logs, project rules, failed approaches, blocking hooks, reusable skills, tests that did not exist yesterday, and scratch notes that say: do not waste another hour here.

That is learning.

It is not training in the usual sense. Nobody updated the weights. But past experience changed future behavior through durable external artifacts. That is enough to make the word "learn" useful, as long as we do not pretend all learning has the same shape.

This is a snapshot, not a product survey. The shift I care about is the move from one-off assistants that answer inside a session to systems that accumulate operational knowledge across sessions, tools, sandboxes, and agents.

The practical question is not whether agents can learn without retraining. They already can.

The practical question is what kind of learning we are willing to let them keep.

## The Learning Outside The Model

Most conversations about AI improvement still collapse back to the model: benchmark movement, base-model capability, post-training recipe. Those things matter, but they miss the part of the system I keep seeing in real work.

A coding agent is not just a model. It is a model sitting inside a small operating environment: a repo, a shell, a filesystem, tests, instructions, permissions, command history, tool outputs, review comments, and whatever memory system the product gives it.

When that environment changes, the agent changes.

If one session discovers that a migration script silently rewrites generated files, the next session can avoid touching those files. If one run adds a failing test, the next run inherits a sharper definition of the problem. If a user keeps correcting the agent for skipping visual QA, that correction can become a project habit.

The workspace starts acting like an external nervous system.

![Agent learning pipeline](/images/AI/agent-learning/agent-learning-pipeline.svg)

The useful part of this diagram is not the exact boxes. It is the direction of compression.

Raw session activity produces traces: prompts, tool calls, command output, failed attempts, user corrections, partial fixes, and successful patterns. Hooks or other observers make those traces durable. They turn behavior that would have died with the session into something the system can inspect later.

Traces are still too noisy to guide behavior directly, so the next step is distillation. Repeated observations become project instincts: lessons that should change behavior in one repository or one workflow. A project instinct might say that visual changes need browser review, generated migrations should never be edited directly, or a particular CI failure usually means the local cache is stale.

That is already different from memory. Memory says, "This may matter later." An instinct says, "Act differently next time."

The dangerous step is promotion. A project lesson can become a global habit. Sometimes that is right. Often it is not. A debugging pattern that saves one stack can damage another. A correction that made sense for one team can annoy a different user. Scope is not bookkeeping; it is part of the safety model.

There is a second step that is easy to miss: evolution. Promotion changes where a lesson applies. Evolution changes what kind of artifact the lesson becomes.

A transcript says what happened. A memory says what may matter later. A rule says what must or must not happen. A hook enforces something outside the model's discretion. A command packages a workflow. A skill turns a repeated pattern into a reusable procedure. A subagent delegates a kind of work to a specialized role.

Putting all of that under the word "memory" makes the system sound softer than it is. Some of these artifacts do not merely remind the agent. They steer it.

## Memory Has Its Own Objective

An agent can finish a task and still write a terrible memory. It can store a temporary workaround as a permanent fact. It can turn one correction into a global preference. It can preserve a mistaken diagnosis because the final answer sounded confident. It can remember too much and make the next session worse.

Bad memory has a different failure mode from a bad answer. A bad answer is local. A bad memory becomes future context.

![Task completion and memory quality are different objectives](/images/AI/agent-learning/memory-objectives.svg)

That is why memory quality needs its own standard.

For task completion, we usually ask whether the change worked. Did the tests pass? Did the user get what they asked for? Did the agent avoid making a mess?

For memory, the question is stranger: did the agent save the smallest thing that will improve future behavior, with the right scope, enough provenance, and the right amount of uncertainty?

Those are related goals, not identical goals. The incentives can even conflict. A completion-seeking agent may write a memory that makes its path look cleaner than it was, summarize away the false start that future sessions need, or store the user's last correction as a permanent rule.

The ideal memory is not the longest memory. It is the smallest durable representation that improves future behavior without adding false confidence.

That definition implies separate evals. A good agent should be judged both on doing the work and on deciding what should persist after the work. The second part needs its own failure labels: false persistence, stale authority, preference overreach, context pollution, contradiction drift, and memory poisoning.

This is where "instinct" is a useful informal term. It is not a formal product category. It just names the middle layer between description and enforcement.

An instinct is a memory that has started shaping behavior, without becoming a hard rule. It lives somewhere between fact and authority. That middle layer is powerful. It is also exactly where overreach happens.

## Dreams Are The Offline Pass

The agent should not be doing all of its memory hygiene in the middle of a task. During a task, it is trying to make progress, read errors, edit files, choose tools, and decide what to do next. That is a bad moment to decide what the system should carry forward for weeks.

The language of "dreams" is useful because it names a separate phase. Anthropic's managed-agent [Dreams documentation](https://platform.claude.com/docs/en/managed-agents/dreams) describes an offline pass over past sessions and memory. A dream reads an existing memory store plus prior session transcripts, then produces a separate output memory store. The input store is not modified directly, so the output can be reviewed, used, or discarded.

That separation is the architectural point.

Incremental memory writes during work are local and opportunistic. A dream is retrospective. It can look across sessions, merge duplicates, replace stale entries, notice contradictions, and surface patterns that were invisible inside a single run.

In other words, dreaming is garbage collection for agent experience.

It is also more than garbage collection. Done carefully, it can promote experience across levels: raw trace to observation, observation to memory, memory to instinct, instinct to rule, command, skill, or agent.

The word "carefully" is doing a lot of work. A system that dreams too aggressively can promote a one-off workaround into permanent policy, confuse one correction with a durable preference, or overfit to the last few sessions and call that learning.

The best dream is not the one that remembers everything. It is the one that throws away the seductive junk.

That is very different from "the agent remembers things." It is closer to maintenance.

## Context Is Not Memory

Longer context windows do not solve this.

A long context window lets the model see more text. That is useful. It is not the same as deciding what deserves to persist. Context is exposure. Memory is selection. Rules are authority. Hooks are enforcement.

Anthropic's [memory tool documentation](https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool) makes this distinction concrete: the tool lets Claude store and retrieve information across conversations through a persistent memory directory, instead of loading everything into the active context window. The point is just-in-time retrieval, not infinite context.

Claude Code's [project memory documentation](https://code.claude.com/docs/en/memory) draws a related boundary. `CLAUDE.md` files are human-written persistent instructions. Auto memory is written by Claude from corrections and preferences. Rules and hooks sit at different levels of force. Memory and instructions are context, not enforced configuration; if something must be blocked regardless of what the model decides, use a hook.

If the agent should remember that a repo uses `pnpm`, memory is fine. If the agent must never deploy without running a smoke test, memory is too weak. If the agent keeps doing a workflow badly, the answer may not be another note. It may be a command, a checklist, a test, a hook, or a smaller tool surface.

The mistake is letting a weak artifact do a strong artifact's job.

Cross-session, cross-agent, and cross-sandbox memory are not the same problem. The same agent improving after yesterday's work is different from one agent sharing lessons with another, or from knowledge surviving disposable sandboxes. The broader the sharing boundary, the more important provenance and scope become.

The memory system has to know not just what it knows, but where that knowledge is allowed to act.

## The World Has To Keep Score

Reset-style loops make this obvious.

The Ralph loop is a useful example: run an agent repeatedly against a stable objective, with each iteration starting fresh but inheriting the changed workspace. Ralph's own description of [The Loop](https://wiggum.dev/concepts/the-loop/) is basically an argument for externalized state: files, tests, git history, progress markers, and the codebase itself carry the work forward.

![The Ralph loop needs the world to keep score](/images/AI/agent-learning/ralph-loop-scorekeeper.svg)

The trick is mundane. The world is keeping score.

If iteration one writes a test, iteration two sees it. If iteration one records a failed approach, iteration two can avoid it. If iteration one leaves a progress file that says exactly what is still unresolved, iteration two does not need to rediscover the whole situation.

Without that scorekeeping, the loop becomes expensive repetition. The agent starts over, rereads the same files, makes the same plan, hits the same blocker, and produces another polished status update.

That is not autonomy. It is amnesia with a nice final answer.

This is the useful way to interpret complaints that a given agent is bad at this kind of loop. Is it rediscovering facts, repeating failed strategies, working against vague acceptance criteria, or burning tokens instead of stopping when blocked? Those are system-quality questions, not only model-quality questions.

The loop works only when progress is externally legible: tasks are explicit, tests are cheap enough to run, blockers are recorded, failed approaches are marked, git history is meaningful, and the agent can tell whether there is less work than before.

The agent only looks persistent when the environment is persistent in the right ways.

## Review The Learned Parts

Once learned artifacts shape future behavior, they become operationally closer to code than chat history.

They may not compile. They may not have unit tests. But they affect what the agent will do next week.

That means they need review.

I do not mean every tiny memory needs a meeting. I mean broad or behavior-changing artifacts need a way to be inspected. What changed? Why did it change? Where does it apply? How confident are we? How do we remove it?

A memory diff should be reviewable. A proposed rule should name its scope. A command should show the workflow it automates. A skill should have examples. A subagent should say when it should be invoked and when it should stay out of the way.

Otherwise the system gets a hidden policy layer.

This is already visible in software work. A model can produce a plan that is useful for continuation while still being poor for human review: enough context for the next agent run, but not enough clarity about the data model, ownership boundary, migration risk, or PR breakdown. When the second PR reveals that the first PR chose the wrong abstraction, that is not just a planning failure. It is a missing fence between model-facing continuation and human-reviewable commitment.

The same fence is needed for agent learning. A memory-facing artifact may help the model continue, but that does not mean it should silently shape future behavior. Mature systems will need reviewable memory, scoped instincts, discardable dreams, and artifact promotion rules that humans can audit.

Agents do not need to train to learn.

They can learn by externalizing experience into memory, consolidating it offline, turning repeated experience into instincts, and executing those instincts through commands, skills, rules, hooks, subagents, and reset-style loops.

But every one of those mechanisms can go wrong. Memory can become stale. Dreams can overfit. Instincts can become rigid. Skills can encode bad workflows. Rules can overconstrain. Loops can repeat failure. Cross-session knowledge can become cross-session contamination.

The next serious leap in agents may not come from larger context windows or longer autonomous runs. It may come from better systems for turning experience into reviewable, scoped, durable behavior.

The central question is not "Can the agent remember?"

The better question is: what should the agent be allowed to learn, how should that learning be represented, and what evidence would prove that the representation improves future behavior?
