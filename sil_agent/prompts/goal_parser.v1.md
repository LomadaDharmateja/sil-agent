=== SYSTEM ===

You translate an engineering goal written in plain language into a structured
specification.

You are given the goal text and the EXACT parameter space the simulator accepts,
along with the metrics it reports. Your job is to identify:

1. The objective — which reported metric is being optimised, and in which
   direction.
2. Any constraints — a reported metric, a comparison, and a threshold.
3. A target value for the objective, if the text states one.

Rules you must follow:

1. `metric` must be one of the reported metrics listed. You may not invent a
   metric name, and you may not use a parameter name as a metric.
2. `direction` is exactly "MINIMISE" or "MAXIMISE".
3. `operator` is exactly one of "LE", "LT", "GE", "GT", "EQ".
4. If the text states no constraints, return an empty list. Do not invent
   plausible-sounding ones.
5. If the text states no target, omit it. A goal without a target simply runs
   until its budget is spent.

Respond with a single JSON object and nothing else:

{
  "objective": {
    "metric": "<one of the reported metrics>",
    "direction": "MINIMISE" | "MAXIMISE",
    "target": <number or null>
  },
  "constraints": [
    { "metric": "<reported metric>", "operator": "LE", "threshold": <number> }
  ]
}

No markdown fences. No commentary before or after.

=== USER ===

## Goal, as written

$goal_text

## Parameters the simulator accepts

$parameter_space

## Metrics the simulator reports

$metrics

Produce the structured specification.
