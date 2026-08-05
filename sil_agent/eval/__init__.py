"""Measurement: scoring runs, executing ablations, and reporting the results.

Nothing in this package proposes candidates or knows what a strategy is beyond
its name. It reads the episodes table and produces numbers. Keeping it separate
from the loop is what allows the same harness to score a random sampler in Phase
2 and a full agent in Phase 4 without either one being scored on its own terms.
"""
