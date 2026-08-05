"""Prompts as versioned files on disk, not f-strings buried in agent code.

A prompt is an input that changes results, so it is treated like one: it lives in
a file, it carries a version, and that version is recorded on every
``CostRecord``. A number in the ablation report can therefore be traced back to
the exact template that produced it.

Phase 11 A/B tests prompt versions against the eval harness. This module is the
bookkeeping that makes that possible; it costs almost nothing now and is
unpleasant to retrofit once results exist that were produced by prompts nobody
kept.

**Templates use ``$name`` substitution, not ``{name}``.** Prompts here contain
JSON examples full of braces, and ``str.format`` would try to interpret every one
of them. ``string.Template`` leaves braces alone.

File format — one file per template, two labelled sections:

    === SYSTEM ===
    You are ...

    === USER ===
    The goal is $goal_text
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path
from string import Template

PROMPT_DIR = Path(__file__).parent

SYSTEM_MARKER = "=== SYSTEM ==="
USER_MARKER = "=== USER ==="


@dataclass(frozen=True)
class PromptTemplate:
    """One versioned prompt, ready to render."""

    name: str
    version: str
    system: str
    user_template: str

    @property
    def identifier(self) -> str:
        return f"{self.name}.{self.version}"

    def render(self, **values: object) -> tuple[str, str]:
        """Fill in the placeholders. Returns (system, user).

        ``substitute`` rather than ``safe_substitute``: a missing value is a bug
        in the caller, and a prompt containing a literal ``$goal_text`` sent to a
        model is a silent failure that produces plausible nonsense.
        """
        rendered = Template(self.user_template).substitute(
            {key: str(value) for key, value in values.items()}
        )
        return self.system, rendered


@cache
def load(name: str, version: str = "v1") -> PromptTemplate:
    """Load ``<name>.<version>.md`` from this directory.

    Cached: templates are read once per process and never change under it.
    """
    path = PROMPT_DIR / f"{name}.{version}.md"
    if not path.exists():
        available = sorted(p.name for p in PROMPT_DIR.glob("*.md"))
        raise FileNotFoundError(f"no prompt {path.name}; available: {', '.join(available)}")

    text = path.read_text(encoding="utf-8")
    if SYSTEM_MARKER not in text or USER_MARKER not in text:
        raise ValueError(f"{path.name} must contain both {SYSTEM_MARKER} and {USER_MARKER}")

    _, remainder = text.split(SYSTEM_MARKER, 1)
    system, user = remainder.split(USER_MARKER, 1)

    return PromptTemplate(
        name=name,
        version=version,
        system=system.strip(),
        user_template=user.strip(),
    )
