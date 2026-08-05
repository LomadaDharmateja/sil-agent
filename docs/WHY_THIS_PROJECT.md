# Why this project exists

Read this first. It's the non-technical document — what the project is, why anyone cares, and how to talk about it in an interview.

---

## 1. The one-sentence version

> An AI system that finds the best settings for an engineering design by proposing options, testing them in a simulator, learning from each result, and explaining its reasoning at every step.

---

## 2. The real-world problem

Engineers work with **simulation models**. You feed in settings, it predicts outcomes:

```
gear ratio 9.2  ┐
motor power 140 kW ├──►  [ simulator ]  ──►  energy 15.8 kWh/100km
vehicle mass 1850 kg ┘                        0–100 km/h in 8.7 s
                                              peak temp 41 °C
```

The engineer's actual job is the inverse: **given targets, find the settings that hit them.**

This is genuinely hard for three reasons:

**The space is enormous.** Ten parameters at twenty plausible values each is 20¹⁰ — about 10 trillion combinations. You cannot try them all.

**Each test is expensive.** A simple analytic model runs in milliseconds, but real crash simulation, CFD, or full vehicle co-simulation takes minutes to hours. You get maybe 50–500 evaluations before someone needs an answer.

**The parameters interact.** Raising gear ratio improves efficiency but hurts acceleration. Increasing motor power fixes acceleration but adds mass, which hurts efficiency again. You can't tune them one at a time.

This has a name in industry: **design space exploration**, or **design of experiments (DoE)**. It's a funded, staffed activity at every company that simulates anything.

---

## 3. Why the two existing approaches both fall short

### An engineer does it by hand

They use physics intuition. They know which parameters interact and which constraint is currently binding. When a result comes back bad, they understand *why* and adjust accordingly.

But: perhaps ten attempts a day. The reasoning lives in their head and is never written down. Results depend heavily on which engineer you assigned.

### An optimisation algorithm does it

Bayesian optimisation, genetic algorithms, Optuna. Thousands of attempts, running overnight, mathematically principled.

But it is **blind**. It has no concept of physics. It doesn't know that gear ratio and motor power interact — it can only infer correlations from samples. It cannot use knowledge that exists in words ("this motor family loses efficiency above 8000 rpm"). And critically, **it cannot explain itself.** It outputs numbers. Ask why it chose them and the honest answer is "the acquisition function was highest there."

### The gap

| | Fast | Uses domain knowledge | Explains itself |
|---|---|---|---|
| Human engineer | ✗ | ✓ | ✓ |
| Bayesian optimisation | ✓ | ✗ | ✗ |
| **This project** | ✓ | ✓ | ✓ |

That third column is not a nice-to-have. In automotive and aerospace, engineering decisions must be **justified and auditable** to be signed off. A black-box optimiser's output often has to be re-justified by hand before anyone will act on it.

---

## 4. Why an LLM is the right tool here — and the honest caveat

An LLM can hold and apply knowledge expressed in language. You can tell it, in the prompt: *"this is a rear-wheel-drive EV, the battery is thermally limited above 40 °C, and the customer cares more about range than 0–100."* No optimisation algorithm can consume that.

It can also produce a **hypothesis with a reason**: *"consumption is dominated by the motor operating off its efficiency peak at cruise, so I'll raise the gear ratio to shift the cruise operating point — accepting that acceleration will worsen."*

**The honest caveat, which you should state in interviews before anyone asks:** an LLM left alone will confidently invent nonsense. That's why this design never lets it grade itself. The simulator computes every number. The LLM proposes and explains; **reality decides.** If it claims an improvement, that claim is checked against a real objective value before anything is recorded.

This is the single most important idea in the project, and it's what separates it from the large pile of "AI agent" demos that just talk to themselves.

---

## 5. Who cares about this

Any organisation running expensive simulations against multi-parameter designs:

- **Automotive** — powertrain sizing, thermal management, control calibration, aero. VW, Bosch, ZF, Continental, Mercedes, BMW.
- **Aerospace** — structures, propulsion
- **Energy** — turbine design, grid and battery sizing
- **Chemical and process** — reactor and plant parameters
- **Also, directly: ML itself** — hyperparameter search is exactly the same problem shape

That last one matters. It means the project is legible to a pure-software AI team *and* to an industrial engineering team. Very few portfolio projects are legible to both.

---

## 6. Your personal connection to it — use this

Your VW thesis work included, in your own words, *"parameter sweeps and performance metrics tracked in MLflow."*

**That is this exact problem, solved manually.** You ran the sweeps. You looked at results. You decided what to try next using your own engineering judgement. MLflow recorded the numbers but captured none of the reasoning.

This gives you an origin story that is true, specific, and impossible to fake:

> "During my thesis at Volkswagen I ran parameter sweeps by hand and tracked them in MLflow. What bothered me was that MLflow recorded every number but none of my reasoning — why I tried that next, what I concluded when it failed. So I built a system that captures the reasoning as a first-class artifact, and then I benchmarked it against Optuna to find out whether the reasoning actually helps."

Interviewers remember stories that come from real frustration. They forget project descriptions.

---

## 7. What it demonstrates — mapped to what job ads ask for

| Job ad phrase | What in this project proves it |
|---|---|
| "LLM / agentic systems" | A real plan-act-critique-replan loop, not a prompt chain |
| "production-grade AI" | Checkpointing, resume, retries, circuit breakers, budget caps |
| "evaluation and guardrails" | Ablation study with baselines, structured output validation |
| "cost optimisation / inference efficiency" | Multi-provider routing, measured €/run, surrogate pre-screening |
| "observability / monitoring" | OpenTelemetry traces of long-horizon runs, cost attribution |
| "MLOps" | Reproducible seeded experiments, CI-gated regression suite |
| "ML / data science" | Surrogate models, statistical comparison with error bars |
| "domain knowledge in engineering" | Vehicle energy simulation, real constraints |
| "software engineering" | Typed interfaces, migrations, tests, Docker, CI |

One codebase, nine boxes ticked. That's why two deep projects beat six shallow ones.

---

## 8. How to explain it in an interview

### The 30-second version (for "tell me about a project")

> "I built a system that optimises engineering designs by putting an LLM in a closed loop with a simulator. It proposes settings, runs the simulation, reads the result, works out why it fell short, and tries again — carrying forward what it learned. The interesting part isn't the LLM, it's that I benchmarked it against random search and Bayesian optimisation across multiple seeds, so I can tell you exactly how much the reasoning loop is worth."

Stop there. Let them ask.

### The 2-minute version (when they do)

Cover these four beats in order:

**1. The problem.** "Design space exploration. Ten parameters, simulations that cost minutes each, so you get maybe a hundred evaluations. Today you either have an engineer do it by hand — smart but slow and undocumented — or an optimiser do it — fast but blind and unexplainable."

**2. The insight.** "An LLM can use knowledge written in words that no optimiser can consume, and it can give a reason for every proposal. But it will also happily hallucinate. So the architecture never lets it grade itself — the simulator computes every number, and the model only proposes and explains."

**3. The engineering.** "Runs go for hundreds of steps, so it checkpoints after every episode and resumes from a crash. There's a budget governor capping tokens, wall-clock and simulator calls — an agent in a loop can burn real money. Every step is traced with OpenTelemetry so I can see cost and objective value per episode."

**4. The result.** "Here's the ablation table." — then show actual numbers.

### The questions you will be asked, and how to answer

**"Why not just use Bayesian optimisation? It's better at this."**

The most important question, and often a test of intellectual honesty. Answer:

> "For pure sample efficiency on a well-defined numeric space, BO is strong and I benchmarked directly against Optuna's TPE — I'll show you the numbers. Where the agent adds value is where BO structurally can't go: constraints and context that exist only in language, and an explanation for every decision. On my benchmarks it [state your actual result]. I'd use BO for a clean numeric problem and this for one where domain context matters and the result has to be justified to a human."

Never claim you beat BO unless your data says so. "I measured and here's where each approach wins" is a far stronger answer than an unsupported win.

**"How do you know the LLM isn't making it up?"**

> "It structurally can't, on the thing that matters. Whether a candidate improved is computed from the simulator's objective value and injected into the critic — the model doesn't get a vote on it. It only explains why. And every parameter it proposes is validated against a schema the simulator declares, so it can't invent a parameter that doesn't exist."

**"What did it cost to run?"**

> "About €X per full run after routing. The planner and critic need a strong model, but extraction and formatting go to a cheap one — that alone cut cost by Y%. Most of my development ran entirely on free tiers across Gemini, Groq and Cerebras."

Cost awareness reads as senior. Most candidates have never thought about it.

**"What would you do differently?"**

Have a real answer. Something like: "I built the agent before the benchmark harness on my first attempt and spent two weeks unable to tell whether changes helped. I rebuilt with baselines first." Self-criticism with a concrete lesson is a strong signal.

**"How would you productionise this?"**

> "Most of it already is — durable state, resumability, budget caps, tracing. What's missing for real deployment is multi-tenancy, a proper job queue with priorities, secrets management for customer simulator credentials, and an approval step before any result feeds a real design decision."

### What not to say

- Don't call it "multi-agent." It's one agent with several roles. Someone will ask you to describe the inter-agent protocol and you'll have nothing.
- Don't say "AI-powered" or "revolutionary." Describe mechanism.
- Don't hide the negative results. If reflection lost to Optuna on some benchmark, say so. Interviewers trust people who report inconvenient findings, and they can smell an unblemished story.
- Don't lead with the framework names. Lead with the problem. Nobody was ever hired for knowing what LangGraph is.

---

## 9. What "done" looks like

You can stop when you have:

1. A public repo with a clear README and an architecture diagram
2. **The ablation table with real numbers and error bars** — the single most valuable artifact
3. A blog post walking through the problem, design, and results
4. A 3-minute demo video
5. Two or three sentences you can say out loud without notes

Item 2 is the one that gets you interviews. Everything else supports it.
