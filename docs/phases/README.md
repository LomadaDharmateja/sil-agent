# Phase logs

One file per phase: `phase-01.md`, `phase-02.md`, and so on. Written **after** the phase is built, while it's fresh.

The point is that in three months you can read these in order and reconstruct exactly how the system was built and why each decision was made — without re-reading the code.

They also do double duty: most of your blog post and half your interview answers come straight out of these files.

---

## Rules

- Written in plain language. If a term is unavoidable, define it once, in-line.
- Honest about what didn't work. A phase log with no problems in it is a phase log nobody will trust, including you.
- Written for you-in-three-months, who will have forgotten everything.
- Every phase log must answer the six questions in the template below.

---

## Template

Copy this for each phase.

```markdown
# Phase N — <short title>

## 1. Why this phase exists
What problem in the overall project does this phase solve?
Why now — why is this the right thing to build at this point rather than later?

## 2. What I built
Plain-language description of what came into existence.
List the files added or changed and what each is responsible for.

## 3. How it works
The mechanism, explained so a competent engineer who has never seen this
codebase could follow it. Include a small worked example with real values
wherever possible.

## 4. Key decisions and trade-offs
What did I choose, what did I reject, and why?
What would have been easier but worse?
What would have been better but too expensive right now?

## 5. What went wrong
Bugs, dead ends, wrong assumptions, things that took three attempts.
This section is mandatory and must not be empty.

## 6. What this unlocks
What can the next phase do that it couldn't before?
What specifically depends on this being in place?

## Numbers
Any measurements taken this phase — timings, costs, success rates,
token counts. Even rough ones. Numbers now save guessing later.

## Interview angle
One or two sentences: what does this phase let me claim, and what
evidence backs the claim?
```

---

## Why the "what went wrong" section is mandatory

Two reasons.

It's the most useful section when you come back to the code, because bugs recur and dead ends get re-explored.

And it's the section that makes interviews go well. "What was the hardest part?" is asked in almost every technical interview, and most candidates improvise something vague. You'll have twelve documented answers with specifics.
