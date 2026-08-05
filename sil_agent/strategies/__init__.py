"""Candidate-proposing strategies. Phase 2 adds grid search, Optuna TPE and the
LLM variants; all of them satisfy the same protocol and run through the same harness."""

from sil_agent.strategies.base import Strategy
from sil_agent.strategies.random_search import RandomSearch

__all__ = ["RandomSearch", "Strategy"]
