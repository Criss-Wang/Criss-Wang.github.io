---
title: "The Missing Fence Between AI Plans And PR Stacks"
excerpt: "AI agents can produce plausible implementation plans, but teams need a human-reviewable artifact before those plans become a stack of PRs."
date: 2026/06/07
categories:
  - Blogs
tags:
  - AI
  - Agents
  - Engineering
  - Code Review
layout: post
toc: true
---

I keep running into a specific failure mode with AI-assisted spec-driven development in existing codebases.

The model can produce a large, plausible plan for a feature. It may understand the product goal. It may identify the right modules. It may break the work into steps. It may even name the files that should change.

But the plan is usually written for one very specific reader: the model itself.

That is fine if the next step is another model call.

It is much less fine if the next step is a team deciding whether the design is actually good.

In a mature codebase, "good" rarely means "the steps sound coherent." It means the data model fits the existing system. It means the abstraction belongs to the right layer. It means the code shape will survive future use cases. It means the UI follows the product's conventions. It means the PR stack can be reviewed without making every reviewer reconstruct the entire plan from scratch.

This is where the current AI coding workflow often feels wrong.

The model gives you a plan. The plan is too big for one clean PR. So the natural instinct is to split it into a stack: PR1 sets up the data model, PR2 adds core logic, PR3 wires up the UI, PR4 handles edge cases, PR5 adds tests and polish.

That sounds reasonable until PR2 reveals that PR1 chose the wrong abstraction.

Now you are not just editing PR2. You are changing the foundation underneath the stack.

That is the PR2 -> PR1 problem.

The expensive part is not that the model was wrong. Plans are always wrong in some way. The expensive part is that the wrongness gets discovered after the team has already started paying implementation and review costs.

You end up with recursive rollback:

- PR2 exposes a flaw in PR1.
- PR1 needs to be redesigned.
- PR2 needs to be rewritten.
- The plan needs to be updated.
- Reviewers have to discard part of the mental model they just built.
- The author loses confidence and asks the model for another giant plan.

This breaks the economic promise of AI-assisted development.

AI is supposed to make iteration cheaper. But if the model accelerates the team into the wrong implementation shape, the cheap generation step creates expensive human cleanup later.

The mistake is treating the model's plan as if it is already a review artifact.

It is not.

## A Model Plan Is Not A Human Plan

A model-facing plan is optimized for continuation.

It preserves context. It lists steps. It names files. It restates assumptions. It keeps optional branches alive because the model may need them later. It stores enough local state so the next agent turn can keep moving without rediscovering everything.

That is a useful object.

But it is not the same as a human-facing engineering plan.

A human-facing plan is optimized for judgment.

It should make it easy to answer different questions:

- What decision is being asked of reviewers?
- What data model change is being proposed?
- Which existing abstractions does this rely on?
- Which team, module, or layer owns the important boundary?
- What would make this approach unacceptable?
- Which later PRs depend on this early decision?
- What is the smallest concrete thing reviewers can inspect before implementation begins?

Those are different compression targets.

The model-facing plan says: "Here is everything I might need in order to continue."

The human-facing plan says: "Here are the few decisions that must be correct before continuing is worth it."

If we do not separate those objects, we ask humans to review the wrong thing. They are forced to read markdown written as agent memory and pretend it is a design doc. Or they review PR1 without enough visibility into why PR2, PR3, and PR4 will depend on it.

That is how a foundational mistake survives until it becomes expensive.

## The Fence

The workflow I want is simple:

1. The model explores the codebase and proposes a feature plan.
2. The model converts that plan into a human-reviewable artifact.
3. Humans review the artifact for project-specific engineering judgment.
4. Only then does the model or author break the work into PRs.

The important part is step 2.

After the model plans the feature, it should not immediately start coding a stack of PRs. It should produce a fence.

By "fence," I mean an intermediate artifact that prevents the team from crossing into implementation until the expensive design choices are visible.

The fence is not bureaucracy for its own sake. It is a way to force alignment while changes are still cheap.

The format can vary.

For some teams, the best fence might be a draft large PR. The model implements the broad shape of the feature in one intentionally unmerged PR. Nobody pretends it is ready. It is a design probe. Reviewers can inspect real code, real types, real queries, real UI structure, and real integration points. Once the design feels right, the work can be broken into smaller PRs.

For other teams, the best fence might be a one-page design note. The model describes the data model, code logic, affected modules, migration path, PR breakdown, risks, and local conventions. Reviewers can align on the shape without reading a pile of generated code.

For some teams, the fence might be a hybrid: a short design note plus one or two code sketches. Not full implementation, but enough concrete shape to review the hard parts.

The exact format matters less than the job it performs.

The fence must turn model context into human engineering alignment.

## What The Fence Should Contain

A useful review fence should answer a small set of questions.

First: what is the proposed data model?

In existing projects, data model choices are often the hardest to unwind. If the model invents the wrong entity, stores state in the wrong place, or ignores an existing invariant, every later PR inherits that mistake.

The fence should make the data model visible before implementation starts.

Second: what is the proposed code shape?

Not every implementation detail needs to be decided. But reviewers need to know the intended ownership boundaries. Is the change going into an existing service, a new helper, a shared utility, a page-level component, or a backend endpoint? Is it following the grain of the codebase, or creating a parallel mini-system because that was easier for the model?

Third: what local conventions matter?

Existing projects are different from greenfield projects. In a greenfield app, the model can often invent a coherent local style. In an existing codebase, coherence means fitting into decisions that already exist. Naming, file organization, component boundaries, test style, migration patterns, permission checks, error handling, logging, and UI density all matter.

The fence should name the conventions the implementation is expected to follow.

Fourth: what is the PR stack, and where are the dependencies?

The PR breakdown should not just be a list of slices. It should identify which PRs are foundational and which PRs are downstream. If PR3 depends on a data structure introduced in PR1, reviewers should know that while reviewing PR1.

This is where the fence directly attacks the PR2 -> PR1 problem.

Fifth: what would cause us to reject the plan?

This is probably the most underrated part. A good review artifact should make rejection cheap. It should say, explicitly, "If this data model is wrong, stop here." Or, "If this module boundary is unacceptable, do not proceed to implementation." Or, "If the UI needs to follow a different pattern, resolve that before code generation."

The goal is not to make the model sound confident. The goal is to surface the decisions where confidence would be dangerous.

## Why Teams Need This More

If I am hacking on a personal project, I can tolerate a lot of bad planning. I can let the model generate too much code, delete half of it, and keep the parts that work. The cost of misunderstanding is mostly mine.

Team codebases are different.

Review is not only a correctness check. It is how a team enforces taste, ownership, maintainability, and shared understanding. A reviewer is not just asking, "Does this work?" They are asking, "Should this code exist in this shape?"

AI-generated plans often skip over that distinction.

They are good at saying what can be done. They are worse at presenting the few choices that need human judgment before the doing begins.

That matters because human review attention is scarce. If the model dumps a long markdown plan full of local reasoning, reviewers will not reliably extract the important decisions. If the model jumps straight into PRs, reviewers may discover those decisions only after they have been encoded in code.

Neither path is ideal.

The fence gives reviewers a better object.

It says: here is the proposed shape of the work; here are the assumptions; here are the project-specific choices; here is the stack; here are the points where later work depends on earlier decisions.

Now review can happen at the right level.

## Match The Fence To The Uncertainty

I do not think this is solved.

I also do not think the answer is "always write a design doc." That is too generic.

The artifact has to match the uncertainty.

If the main uncertainty is the data model, the fence might be a schema proposal plus example reads and writes.

If the uncertainty is UI fit, the fence might be screenshots or a prototype.

If the uncertainty is integration risk, the fence might be a draft PR that touches the real interfaces.

The useful rule is:

> Before an AI agent turns a large plan into a PR stack, require a human-reviewable artifact for the highest-cost-to-reverse decision.

That decision may be different every time.

Sometimes it is the schema. Sometimes it is the API. Sometimes it is the component structure. Sometimes it is the migration plan. Sometimes it is the styling system.

The point is to identify it before PR1 quietly commits the team to it.

## A Better Agent Loop

The current loop often looks like this:

1. Ask the model to plan.
2. Ask the model to implement.
3. Review the first PR.
4. Discover that the plan had the wrong shape.
5. Ask the model to repair the stack.

The better loop looks like this:

1. Ask the model to plan.
2. Ask the model to produce the review fence.
3. Review the fence.
4. Revise the fence until the core decisions are acceptable.
5. Ask the model to implement PR1 with the approved constraints.
6. Continue the stack with the fence as shared context.

This does two useful things.

First, it gives the model better instructions. The implementation is no longer based on a vague giant plan. It is based on a reviewed artifact that names the important constraints.

Second, it gives reviewers a durable reference. When PR3 arrives, the reviewer can ask whether it still follows the agreed shape instead of reconstructing the entire design from scattered comments.

That is the real leverage: not more planning, but better placement of review.

## What I Would Ask The Model To Produce

For a large feature in an existing codebase, I would experiment with asking the model for something like this before implementation:

> Convert your implementation plan into a human-reviewable design artifact. Do not optimize for your own continuation. Optimize for a senior engineer deciding whether this should be implemented in this codebase. Identify the data model, code boundaries, local conventions, PR stack, dependencies between PRs, risks, and the earliest decisions that would be expensive to reverse. Keep it short enough to review.

That prompt is not magic. The model can still produce something too long, too generic, or too confident.

But it changes the target. The model is no longer being asked to create memory for itself. It is being asked to create alignment for humans.

That distinction matters.

## The Larger Pattern

This connects to a broader problem with agents.

Agents are increasingly good at producing internal working context. They can write plans, todos, ledgers, scratch files, and implementation notes. Those artifacts can be useful for task completion.

But task-completion artifacts and review artifacts are not the same thing.

The agent needs memory to keep moving.

The team needs judgment surfaces to decide whether movement is good.

If we confuse those two needs, we get generated process that feels productive but does not make the work easier to review.

The missing fence between AI plans and PR stacks is one example of that mismatch.

The model thinks it has planned. The author thinks the work is ready to slice. The reviewer sees a PR and has to infer the design backwards.

That is backwards.

The design should become reviewable before the stack begins.

## Conclusion

AI-assisted development should not jump directly from model plan to PR stack in an existing codebase.

The plan needs to pass through a human-reviewable fence first. That fence might be a draft large PR, a one-page design note, a code sketch, a prototype, or some other artifact. The format is negotiable. The function is not.

It must expose the expensive decisions before implementation makes them expensive to change.

That is how we prevent PR2 from forcing a redesign of PR1. More importantly, it is how we make AI-generated work legible to the humans who still own the codebase.

The future of AI coding in teams probably depends less on bigger plans and more on better review surfaces.
