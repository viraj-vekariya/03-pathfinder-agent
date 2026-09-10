"""Turn a dependency graph into an install plan, and grade a plan against ground truth.

This module is the ARBITER of the whole evaluation. The agent proposes an install
order; this decides whether it is right. So its correctness matters more than anything
the agent does, and every check here is a property of the plan rather than a comparison
against one blessed answer - because a graph normally has many valid topological orders
and grading against one of them would fail correct plans.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .algorithms import (CycleError, Graph, depth_map, detect_cycles,
                         install_order, longest_path_dag)


@dataclass
class Plan:
    """A proposed install order plus everything needed to explain it."""

    root: str
    order: List[str]
    depth: int
    closure_size: int
    cycles: List[List[str]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {"root": self.root, "order": self.order, "depth": self.depth,
                "closure_size": self.closure_size, "cycles": self.cycles}


@dataclass
class Grade:
    """The verdict on a proposed plan.

    `valid` is the headline: a plan is valid when every package appears exactly once
    and no package is installed before something it depends on. Everything else is
    diagnostic, so a failure says WHY rather than just "wrong".
    """

    valid: bool
    coverage: float                 # fraction of the required closure that appears
    order_violations: int           # dependency installed after its dependent
    missing: List[str] = field(default_factory=list)
    extra: List[str] = field(default_factory=list)
    duplicates: List[str] = field(default_factory=list)
    first_violation: Optional[Tuple[str, str]] = None
    reason: str = ""

    def as_dict(self) -> Dict[str, object]:
        return {"valid": self.valid, "coverage": round(self.coverage, 4),
                "order_violations": self.order_violations,
                "missing": self.missing[:10], "extra": self.extra[:10],
                "duplicates": self.duplicates[:10],
                "first_violation": list(self.first_violation) if self.first_violation else None,
                "reason": self.reason}


def build_plan(graph: Graph, root: str) -> Plan:
    """The ground-truth plan for one package."""
    depth, _ = longest_path_dag(graph, root)
    closure = graph.subgraph_reachable_from([root])
    cycles = detect_cycles(closure)
    try:
        order = install_order(graph, [root])
    except CycleError:
        # A cyclic closure has no total order. Rather than failing, report the order of
        # the acyclic part - which is what a real resolver does before asking a human
        # to break the cycle. Airflow's own package family is genuinely cyclic, so this
        # branch is exercised by real data, not a synthetic case.
        order = []
    return Plan(root=root, order=order, depth=depth,
                closure_size=len(closure.nodes), cycles=cycles)


def required_closure(graph: Graph, root: str) -> Set[str]:
    return set(graph.subgraph_reachable_from([root]).nodes)


def grade_plan(graph: Graph, root: str, proposed: Sequence[str]) -> Grade:
    """Grade a proposed order by its PROPERTIES, not against one blessed answer.

    A dependency graph normally admits many valid topological orders. Comparing against
    a single reference would mark correct plans wrong and make the evaluation measure
    agreement-with-my-tie-break rather than correctness.

    Four checks, in the order a reader would ask them:
      1. every required package present  (coverage)
      2. nothing invented                (extra)
      3. nothing listed twice            (duplicates)
      4. no package before its dependency (order_violations)
    """
    required = required_closure(graph, root)
    proposed = [p for p in proposed if p]

    seen: Set[str] = set()
    duplicates: List[str] = []
    for package in proposed:
        if package in seen:
            duplicates.append(package)
        seen.add(package)

    missing = sorted(required - seen)
    extra = sorted(seen - required)
    coverage = len(required & seen) / len(required) if required else 1.0

    position = {package: i for i, package in enumerate(proposed)}
    violations = 0
    first: Optional[Tuple[str, str]] = None
    for package in proposed:
        for dependency in graph.neighbours(package):
            if dependency not in position:
                continue        # absent dependencies are counted by `missing`, not here
            if position[dependency] > position[package]:
                violations += 1
                if first is None:
                    first = (package, dependency)

    valid = (not missing and not duplicates and violations == 0)

    if valid:
        reason = f"valid: {len(proposed)} packages, dependencies before dependents"
    elif missing:
        reason = f"missing {len(missing)} required package(s), first: {missing[0]}"
    elif duplicates:
        reason = f"{len(duplicates)} duplicate entr(ies), first: {duplicates[0]}"
    else:
        reason = (f"{violations} ordering violation(s), first: "
                  f"{first[0]} installed before its dependency {first[1]}")

    return Grade(valid=valid, coverage=coverage, order_violations=violations,
                 missing=missing, extra=extra, duplicates=duplicates,
                 first_violation=first, reason=reason)


def packages_by_depth(graph: Graph, min_closure: int = 2
                      ) -> Dict[int, List[Tuple[str, int]]]:
    """Group every package by the depth of its dependency tree.

    This is what makes the depth sweep possible: to ask how plan quality varies with
    depth, you need real tasks at each depth. Packages with a closure of one (no
    dependencies at all) are excluded - "install X" where X needs nothing is not a
    planning problem and would inflate accuracy at depth 0.
    """
    buckets: Dict[int, List[Tuple[str, int]]] = {}
    for package in sorted(graph.nodes):
        try:
            depth, _ = longest_path_dag(graph, package)
            closure = len(graph.subgraph_reachable_from([package]).nodes)
        except CycleError:
            continue           # no well-defined depth; excluded from the sweep
        if closure < min_closure:
            continue
        buckets.setdefault(depth, []).append((package, closure))
    for depth in buckets:
        buckets[depth].sort(key=lambda pair: pair[1])
    return buckets


def describe_closure(graph: Graph, root: str, metadata: Dict[str, Dict] | None = None,
                     max_packages: int = 200) -> str:
    """Render a closure as the text an agent is given.

    This is the function that decides how many tokens a task costs, which makes it the
    lever behind the whole context-truncation finding. It is deliberately compact -
    one line per package - because a verbose format would hit the model's context limit
    at a shallower depth and make the effect look worse than it is.
    """
    metadata = metadata or {}
    closure = graph.subgraph_reachable_from([root])
    levels = depth_map(graph, root)
    packages = sorted(closure.nodes, key=lambda p: (levels.get(p, 99), p))[:max_packages]

    lines = []
    for package in packages:
        deps = sorted(graph.neighbours(package))
        requires = ", ".join(deps) if deps else "nothing"
        lines.append(f"{package} requires {requires}")
    return "\n".join(lines)
