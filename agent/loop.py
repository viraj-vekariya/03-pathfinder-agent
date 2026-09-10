"""The plan -> act -> observe loop.

The agent is asked one question: given a package, produce a valid install order. It has
tools that can answer that exactly. The evaluation asks whether it manages to.

Structure of one run:

    1. PLAN     the model chooses a tool, given the task and what it has learned so far
    2. ACT      the tool runs - real graph algorithms, real retrieval
    3. OBSERVE  the result is summarised into bounded memory
    ... repeat up to max_steps ...
    4. ANSWER   the model emits the final ordered list

Every prompt is measured before it is sent, and if it exceeds the model's context the
run RECORDS that rather than letting the tokenizer quietly drop the tail. That record is
the entire mechanism behind the depth finding: the agent does not get worse at
reasoning as depth grows, it gets less of the problem.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from graph.algorithms import Graph
from graph.planner import Grade, grade_plan, required_closure

from .llm import BaseLLM, Completion
from .memory import Memory
from .tools import Toolbox, ToolResult

PLAN_PROMPT = """You are resolving Python package installation order.

Task: list every package needed to install {package}, in an order where each package
appears after everything it depends on.

Tools:
{tools}

What you have learned so far:
{memory}

Reply with exactly one tool call, like: correct_install_order({package})
Tool call:"""

# The answer prompt is deliberately plainer than the plan prompt, and the difference
# was measured rather than guessed. Three variants were tried against flan-t5-base on
# the same facts:
#   "Facts:\n{memory}\n\nQuestion: list every package..."   -> dropped 6 of 9 packages
#   "{memory}\n\nCopy the list of package names..."          -> echoed the lead-in text
#   "Read this: {memory}\n\nWhat is the install order? ..."  -> exactly correct on
#                                                                 flask, 0.12 overall
# A later sweep showed why that last one was misleading: it works when the list is
# short, and collapses to echoing the package name when the prompt names the package.
# A fourth variant, measured across 8 packages:
#   "Read this: ... What is the install order for {pkg}?"  -> validity 0.12, cov 0.61
#   "List every package name in the sentence above"        -> validity 0.12, cov 0.80
#   "{bare list}\n\nRepeat the list above exactly"          -> validity 0.75, cov 0.95
#   "Extract the complete list. Do not stop early."        -> validity 0.12, cov 0.81
# The winner does not mention the package at all. Naming it invites the model to answer
# the question extractively with the shortest span that fits - the name itself. The
# memory is also rendered WITHOUT the "[0] tool:" prefixes: with them, the model's most
# common output was the literal string "[0]".
ANSWER_PROMPT = """{memory}

Repeat the list above exactly, separated by commas."""

_TOOL_CALL_RE = re.compile(r"([a-z_]+)\s*\(\s*([^)]*)\s*\)")


@dataclass
class Step:
    index: int
    kind: str                    # "plan" | "answer"
    prompt_tokens: int
    truncated: bool
    tokens_dropped: int
    raw: str
    tool: Optional[str] = None
    tool_ok: Optional[bool] = None
    latency_ms: float = 0.0


@dataclass
class RunResult:
    package: str
    proposed: List[str]
    grade: Grade
    steps: List[Step]
    memory_stats: Dict[str, object]
    tools_used: List[str]
    any_truncation: bool
    max_prompt_tokens: int
    total_tokens_dropped: int
    context_limit: int
    seconds: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "package": self.package,
            "proposed_length": len(self.proposed),
            "grade": self.grade.as_dict(),
            "steps": len(self.steps),
            "tools_used": self.tools_used,
            "any_truncation": self.any_truncation,
            "max_prompt_tokens": self.max_prompt_tokens,
            "total_tokens_dropped": self.total_tokens_dropped,
            "context_limit": self.context_limit,
            "memory": self.memory_stats,
            "seconds": round(self.seconds, 3),
        }


def parse_tool_call(text: str, known: Dict[str, object]) -> Optional[Tuple[str, List[str]]]:
    """Pull a tool call out of free-form model output.

    Small models do not reliably emit clean syntax, so this is forgiving: it scans for
    any `name(args)` whose name is a real tool. Being forgiving here matters because the
    experiment is about context, not about instruction-following - a parser that
    rejected slightly-malformed calls would confound the two.
    """
    for match in _TOOL_CALL_RE.finditer(text.lower()):
        name = match.group(1).strip()
        if name in known:
            args = [a.strip().strip("'\"") for a in match.group(2).split(",") if a.strip()]
            return name, args
    # No syntax at all: fall back to a bare tool name if one appears.
    lowered = text.lower()
    for name in known:
        if name in lowered:
            return name, []
    return None


def parse_package_list(text: str, valid: set[str]) -> List[str]:
    """Extract an ordered package list from free-form output.

    Filtered against the real package set, because a model asked for package names will
    happily produce prose, and counting "the" as a package would flatter every score.
    Order and duplicates are preserved - the grader needs to see them.
    """
    candidates = re.findall(r"[A-Za-z0-9][A-Za-z0-9._-]{1,50}", text)
    out = []
    for token in candidates:
        norm = re.sub(r"[-_.]+", "-", token.lower())
        if norm in valid:
            out.append(norm)
    return out


class PathfinderAgent:
    def __init__(self, graph: Graph, llm: BaseLLM, toolbox: Toolbox,
                 max_steps: int = 4, memory_fraction: float = 0.55):
        self.graph = graph
        self.llm = llm
        self.toolbox = toolbox
        self.max_steps = max_steps
        # Memory gets a FRACTION of the context, not all of it: the instructions, the
        # tool list and the model's own answer all need room. Budgeting 100% to memory
        # guarantees the instructions get truncated, which is the worst possible thing
        # to lose.
        self.memory_budget = max(64, int(llm.context_limit * memory_fraction))

    def run(self, package: str) -> RunResult:
        started = time.perf_counter()
        registry = self.toolbox.registry()
        memory = Memory(self.memory_budget, self.llm.count_tokens)
        steps: List[Step] = []

        for index in range(self.max_steps):
            prompt = PLAN_PROMPT.format(package=package,
                                        tools=self.toolbox.describe(),
                                        memory=memory.render() or "(nothing yet)")
            truncated, dropped = self.llm.would_truncate(prompt)
            completion = self.llm.complete(prompt, max_new_tokens=48)

            call = parse_tool_call(completion.text, registry)
            step = Step(index, "plan", completion.prompt_tokens, truncated, dropped,
                        completion.text[:200], latency_ms=completion.latency_ms)

            if call is None:
                steps.append(step)
                break

            name, args = call
            if not args:
                args = [package]
            try:
                result: ToolResult = registry[name](*args[: _arity(name)])
            except Exception as exc:             # noqa: BLE001
                result = ToolResult(name, False, f"tool error: {exc}", {})

            step.tool, step.tool_ok = name, result.ok
            steps.append(step)
            memory.add(index, name, result.summary, result.answer_text)

            # Once the exact answer is in memory, more tool calls can only push it out
            # of the budget. Stopping is the correct action, not laziness.
            if name == "correct_install_order" and result.ok and not result.truncated:
                break

        answer_prompt = ANSWER_PROMPT.format(
            package=package, memory=memory.render_plain() or "(no information)")
        truncated, dropped = self.llm.would_truncate(answer_prompt)
        completion = self.llm.complete(answer_prompt, max_new_tokens=320)
        steps.append(Step(len(steps), "answer", completion.prompt_tokens, truncated,
                          dropped, completion.text[:400],
                          latency_ms=completion.latency_ms))

        proposed = parse_package_list(completion.text, set(self.graph.nodes))
        grade = grade_plan(self.graph, package, proposed)

        return RunResult(
            package=package,
            proposed=proposed,
            grade=grade,
            steps=steps,
            memory_stats=memory.stats(),
            tools_used=[s.tool for s in steps if s.tool],
            any_truncation=any(s.truncated for s in steps),
            max_prompt_tokens=max((s.prompt_tokens for s in steps), default=0),
            total_tokens_dropped=sum(s.tokens_dropped for s in steps),
            context_limit=self.llm.context_limit,
            seconds=time.perf_counter() - started,
        )


def _arity(tool: str) -> int:
    return 2 if tool == "shortest_dependency_path" else 1
