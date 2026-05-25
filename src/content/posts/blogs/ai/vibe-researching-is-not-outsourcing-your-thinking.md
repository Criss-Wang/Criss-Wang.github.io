---
title: "Vibe Researching Is Not Outsourcing Your Thinking"
excerpt: "Claude made parts of a multi-agent RL project faster, but the hard part stayed mine: deciding where agents synchronize and what claims were earned."
date: 2026/05/07
categories:
  - Blogs
tags:
  - AI
  - Research
  - Agents
layout: post
toc: true
---

# Vibe Researching Is Not Outsourcing Your Thinking

I started to understand "vibe researching" while working on a multi-agent reinforcement learning project.

The problem was not vague. I needed a rollout design for agents that did not all act at the same neat timestep. Some agents waited. Some reacted to changes in the environment. Some needed information from other agents before moving forward. The hard part was not writing `asyncio` code. The hard part was deciding where the system was allowed to synchronize.

Claude was in the loop for a lot of this.

It read papers faster than I could. It compared framework choices. It helped me sketch execution flows, refactor experimental code, debug async behavior, and later draft parts of the paper once the argument had a spine. If I describe it that way, it sounds almost like outsourcing.

It was not.

The useful version of vibe researching is not "ask the model for a research idea and believe the answer." That is just delegation without taste. The useful version is closer to having a fast second mind around the edges of the work. It can bring material closer. It can name assumptions. It can make vague pressure points discussable.

But it does not own the question.

## Where Claude Helped

One useful moment came from execution-flow design.

Claude separated possible agent behavior into periodic ticking and reactive triggering. That was not the algorithm, and I would not call it the contribution. But it gave me a handle. I could ask, for each agent, whether it should act because time had advanced or because something in the environment had changed.

That axis made the system easier to reason about.

Another useful moment came from debugging the async design. Claude kept pushing toward implementation details, but the real question underneath was conceptual: where is the synchronization point? At what moment do agents exchange information, commit actions, observe the next state, and become allowed to move forward?

Once the problem was phrased that way, the code became less mysterious.

That is when AI collaboration feels good to me. Not when it hands down an answer, but when it helps reveal the shape of the thing I was already trying to understand.

## Fluent Momentum

The dangerous part is that the same fluency can carry a wrong assumption farther than it deserves.

If Claude misunderstood the environment model, it could still produce a clean plan. If it assumed synchronous execution when the real system was asynchronous, the plan still looked rigorous for a few paragraphs. If it overstated a contribution in the paper draft, the sentence often sounded better than the claim deserved.

That is the failure mode I keep coming back to: fluent momentum.

The model keeps going. It writes the next paragraph. It fills the missing table. It explains the wrong premise more clearly. If I am tired, the polish can become persuasive. I start reviewing prose instead of the assumption.

The fix was not a better prompt so much as a different relationship. I had to interrupt it. What assumption are you making here? What would make this fail? Are you answering a production-engineering problem when I am still trying to preserve a research experiment?

Those questions changed the role of the assistant. Less oracle. More intern whose work can be useful, but only after review.

## Research Code Has A Different Center

One friction surprised me: Claude often wanted to make the research code too clean too early.

It leaned toward production structure, generalized abstractions, tests, and careful engineering rituals. Those instincts are often good. But early research code is not a product. It is closer to a lab bench. The point is not to make every component elegant. The point is to make the central uncertainty testable.

I had to teach the assistant that local standard.

Prototype first. Preserve the experiment. Avoid stabilizing abstractions before the research shape is clear. In the exploration phase, good code answers the question. Later, if the answer survives, good code makes it reliable.

Confusing those phases is expensive.

## The Part Outside The Chat

The deepest part did not happen in the conversation.

For the core rollout flow, I had to close the chat and sit with the system. I drew the entities, moved imaginary actions through the episode, and kept asking what each component actually knew at each moment. Agents, environment state, pending actions, capabilities, synchronization, data movement: the design only became real after I simulated it privately for a while.

That took a couple of intense days.

This is an underrated limit of AI collaboration: conversation has an interaction cost. When an idea is fragile, explaining it too early can break the line of thought. The model's response gives you another object to evaluate, and evaluation is not free.

Sometimes the best use of the assistant is to prepare the solitude, not replace it.

Claude helped gather material, expose assumptions, clean up code, and make paper sections easier to draft. But the backbone had to exist first. When I asked it to write too early, the result sounded like a paper without knowing what the paper had earned.

The workflow that helped was more specific: here is the claim, here is the evidence, here is what we should not overstate, here is the sentence that must survive. Now help me make it precise.

That is the relationship I trust.

AI can make research faster. It can make the messy middle less lonely. It can help a researcher see more material and test more formulations.

But the researcher still owns the question.
