"""Simulators — the oracle. Entirely LLM-free, which is what lets non-LLM
baselines run through the identical harness."""

from sil_agent.simulators.base import EvalCost, Simulator
from sil_agent.simulators.toy import BENCHMARKS, ToySimulator

__all__ = ["BENCHMARKS", "EvalCost", "Simulator", "ToySimulator"]
