---
title: "The Missing Fence Between AI Plans And PR Stacks"
excerpt: "AI agents can produce plausible plans quickly, but teams need a review surface before those plans harden into a stack of PRs."
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

There is a moment in AI-assisted coding when the work still feels cheap.

The model has read the repo. It has produced a plan. The plan is long, confident, and full of file names. It knows there should be a data model change, a service update, a UI pass, tests, maybe a migration. It has the rhythm of engineering work.

Then the feature is too large for one PR, so the plan gets sliced.

PR1 lays the foundation. PR2 builds on it. PR3 wires the UI. PR4 handles edge cases. PR5 cleans up tests and polish.

On paper, this is exactly what a responsible team wants. Small PRs. Reviewable chunks. A stack instead of a giant dump.

Then PR2 exposes that PR1 chose the wrong abstraction.

Now the cheap part is over.

You are not only editing PR2. You are reopening PR1. The plan needs to change. The reviewer has to throw away part of the mental model they just built. The author starts wondering whether the stack should be collapsed, rebased, or handed back to the model for another heroic rewrite.

I keep thinking of this as the PR2 -> PR1 problem.

The model being wrong is not the surprising part. Plans are always wrong somewhere. The expensive part is where the wrongness is discovered. If the team discovers it after the first foundation PR has already become real, the mistake has been converted from a cheap design question into a stack-management problem.

This is one way AI can quietly make engineering more expensive while looking productive.

It speeds up the path into implementation before the team has agreed on the shape of the work.

## A Plan Is Not A Review Surface

The mistake is treating the model's plan as if it is already a human review artifact.

It usually is not.

A model-facing plan is optimized for continuation. It preserves context. It names files. It keeps branches alive. It restates assumptions. It says enough for the next model call to keep moving without rediscovering the repo.

That is useful. It is just not the same thing as a plan a senior engineer can review.

Human review is not asking, "Can this be implemented?"

Most plausible plans can be implemented. That is not the bar.

The better question is: should this code exist in this shape?

That question is much harder. It asks whether the data model fits the existing system. Whether the ownership boundary is in the right layer. Whether this new helper is really a helper or the beginning of a parallel subsystem. Whether the UI follows the product's density and behavior conventions. Whether PR1 is quietly committing the team to decisions that will not be visible until PR3.

Model plans tend to hide those questions inside orderly prose.

They say "add a shared utility" when the real question is whether the logic belongs in the service layer. They say "create a component" when the real question is whether the existing page pattern should be extended instead. They say "add persistence" when the real question is whether the entity should exist at all.

The plan looks complete because it contains steps.

But steps are not judgment.

## The Fence

What I want between the plan and the PR stack is a fence.

Not a process monument. Not a design-doc ritual for its own sake. A small barrier that prevents the team from crossing into implementation until the expensive decisions are visible.

The workflow would be simple:

1. Let the agent explore and produce its internal plan.
2. Convert that plan into a human-reviewable artifact.
3. Review the artifact for the few decisions that would be painful to reverse.
4. Only then slice the work into PRs.

The second step is the one I usually see missing.

After the model plans, it wants to code. The user also wants it to code, because that is where the tool feels impressive. But in an existing codebase, speed into code is not always progress. Sometimes it is just a faster way to encode the wrong assumption.

The fence forces a pause at the right level.

It asks: what are we about to commit ourselves to?

That artifact can take different forms. For some teams, it might be a one-page design note. For others, a draft PR that nobody pretends is mergeable. For a UI-heavy change, it might be screenshots or a prototype. For a backend change, it might be a schema sketch plus example reads and writes. For an integration-heavy feature, it might be a code probe that touches the real interfaces and stops there.

The format is negotiable.

The function is not.

The fence must turn the model's working context into something humans can judge.

## What Reviewers Need To See Early

The highest-risk decision is not always the same.

Sometimes it is the data model. That is the classic one. If the model invents the wrong entity or stores state in the wrong place, the rest of the stack inherits the mistake. You do not want to discover that in PR3.

Sometimes it is the module boundary. The generated plan may put logic into a shared helper because that is easy to explain, while the codebase actually wants it owned by a service, a route, or a domain object.

Sometimes it is the UI pattern. The model may produce a perfectly reasonable interface that belongs to a different product. It may be too card-heavy, too sparse, too modal-driven, too cheerful, too slow to scan. The issue is not that it fails to render. The issue is that it does not belong.

Sometimes it is the PR stack itself. A stack can look clean while hiding a dependency problem. If PR4 only makes sense if PR1's abstraction survives untouched, reviewers should know that while reviewing PR1.

A useful fence makes these dependencies explicit.

It does not need to explain everything. In fact, it should not. The point is to expose the few decisions where being wrong would be expensive.

I would rather read a short artifact that says:

> If this data model is wrong, stop here. PR2 and PR3 depend on it.

than a beautiful eight-step implementation plan that buries the same fact in the middle.

Good review artifacts make rejection cheap.

That sounds negative, but it is the whole point. If the team is going to reject the data model, reject it before code generation has turned it into migrations, types, UI state, test fixtures, and reviewer fatigue.

## Team Codebases Are Different

On a personal project, I can tolerate a lot of bad AI planning.

I can ask for too much code, delete half of it, keep one useful function, and move on. The cost is mostly mine. If I make a mess, I own the mess.

Team codebases do not work that way.

Review is how a team preserves taste. It is where ownership gets enforced. It is where hidden coupling is noticed. It is where someone says, "This works, but it is not the shape we want here."

That kind of review is already expensive. AI can either make it easier by surfacing the right decisions earlier, or harder by producing a lot of plausible work that reviewers have to reverse-engineer.

The second version is what worries me.

The author thinks the model has planned. The model thinks implementation is the next natural step. The reviewer sees PR1 and has to infer the design backwards. By the time the reviewer understands where the stack is going, the stack has already started.

That is backwards.

The design should become reviewable before the stack begins.

## Match The Artifact To The Risk

I do not think the answer is "always write a design doc."

That advice is too generic, and generic process is how teams end up with documents nobody trusts. The fence should match the uncertainty.

If the risk is schema shape, write the schema proposal and show example operations.

If the risk is UI fit, produce screenshots or a narrow prototype.

If the risk is integration, make a draft PR that touches the real boundary and stops before full implementation.

If the risk is ownership, write down which layer owns what and which existing abstractions are being extended.

The useful rule is:

> Before an AI plan becomes a PR stack, make the highest-cost-to-reverse decision reviewable.

That is enough.

The artifact does not need to be long. It does not need to sound impressive. It needs to make the dangerous assumption visible.

Once the fence is approved, it becomes useful context for the agent too. PR1 is no longer based on a giant plan with a lot of loose branches. It is based on a reviewed constraint. PR3 can be reviewed against the same reference instead of forcing everyone to reconstruct the original intent from comments.

The leverage is not more planning.

It is better placement of review.

## The Prompt I Would Actually Use

For a large feature in an existing codebase, I would not ask the agent to go straight from plan to implementation. I would ask for the fence explicitly:

> Convert your implementation plan into a human-reviewable design artifact. Do not optimize for your own continuation. Optimize for a senior engineer deciding whether this belongs in this codebase. Identify the data model, ownership boundaries, local conventions, PR stack, dependencies between PRs, risks, and the earliest decisions that would be expensive to reverse. Keep it short enough to review.

The prompt will not save you by itself. The model can still make the artifact too generic. It can still sound more certain than it should. It can still miss the old scar in the codebase that every human reviewer remembers.

But it changes the target.

The model is no longer writing memory for itself. It is writing a judgment surface for humans.

That distinction is where a lot of AI coding workflow still feels immature to me. Agents are getting better at producing internal working context: plans, todos, ledgers, scratch files, implementation notes. Those artifacts help the agent keep moving.

But teams do not only need movement.

They need to decide whether the movement is good.

The missing fence between AI plans and PR stacks is one example of that mismatch. The model thinks it has planned. The author thinks the work is ready to slice. The reviewer receives a PR and has to recover the design from the implementation.

We can do better than that.

Before the stack begins, make the expensive decision visible.

That is how AI-assisted development becomes easier to review instead of merely faster to generate.
