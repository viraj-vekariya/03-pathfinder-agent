"""Working memory for one agent run, with an explicit token budget.

The budget is the point. An agent that appends every observation to its prompt will
eventually exceed the model's context, and what happens then is silent: the tokenizer
drops the tail, the model answers about information it never saw, and the failure looks
like bad reasoning rather than lost input.

So memory here is bounded and it EVICTS DELIBERATELY, newest-first-kept, and it records
what it dropped. eval/truncation.py reads those records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


@dataclass
class Observation:
    step: int
    tool: str
    summary: str
    tokens: int
    answer_text: str = ""

    def __post_init__(self) -> None:
        if not self.answer_text:
            self.answer_text = self.summary


@dataclass
class Memory:
    budget_tokens: int
    count_tokens: Callable[[str], int]
    observations: List[Observation] = field(default_factory=list)
    evicted: List[Observation] = field(default_factory=list)

    def add(self, step: int, tool: str, summary: str, answer_text: str = "") -> None:
        obs = Observation(step, tool, summary, self.count_tokens(summary),
                          answer_text or summary)
        self.observations.append(obs)
        self._evict()

    def _evict(self) -> None:
        """Drop the OLDEST observations until the budget is met.

        Oldest-first because in a dependency-resolution loop the most recent tool
        output is the one the next decision depends on. Dropping newest would be the
        one policy guaranteed to remove exactly what is needed.
        """
        while self.used_tokens() > self.budget_tokens and len(self.observations) > 1:
            self.evicted.append(self.observations.pop(0))

    def used_tokens(self) -> int:
        return sum(o.tokens for o in self.observations)

    def render(self) -> str:
        """With step and tool labels. Used for the PLANNING prompt, where the model
        needs to know which tools it has already called so it does not repeat one."""
        return "\n".join(f"[{o.step}] {o.tool}: {o.summary}" for o in self.observations)

    def render_plain(self) -> str:
        """Content only, no step or tool labels. Used for the ANSWERING prompt.

        The labels actively hurt there: asked to answer from memory rendered with
        "[0] tool: ..." prefixes, flan-t5-base's single most common output was the
        literal string "[0]" - it copied the prefix rather than reading past it.
        """
        return " ".join(o.answer_text for o in self.observations)

    def stats(self) -> Dict[str, object]:
        return {
            "kept": len(self.observations),
            "evicted": len(self.evicted),
            "used_tokens": self.used_tokens(),
            "budget_tokens": self.budget_tokens,
            "evicted_tokens": sum(o.tokens for o in self.evicted),
            "over_budget": self.used_tokens() > self.budget_tokens,
        }
