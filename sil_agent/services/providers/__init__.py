"""Provider adapters — the only code in the project that speaks HTTP to an LLM.

``CLAUDE.md``: *All access goes through `services/router.py`. Never call a
provider SDK directly from agent code.* This package is what the router calls,
and nothing above the router imports it.

That boundary is what makes Phase 10's plan — distilling the planner into a
local Qwen3-4B — a new file in here rather than a change to the agent.
"""

from sil_agent.services.providers.base import Completion, Provider

__all__ = ["Completion", "Provider"]
