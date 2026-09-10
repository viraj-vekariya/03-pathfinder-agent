"""Tools the agent can call. Every one is a real graph algorithm or a real retrieval.

Two rules shaped this file:

1. **A tool returns a SUMMARY, not a data structure.** The agent's only channel is its
   context window, so a tool that returns 400 package names has spent the whole budget
   on one call. Each tool decides what is worth the tokens and says so.

2. **Tools cannot lie by omission.** When a result is truncated the summary SAYS it was
   truncated and by how much. A tool that silently returns the first 20 of 200 items
   teaches the agent that 20 is the answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from graph.algorithms import (CycleError, Graph, depth_map, detect_cycles,
                              dijkstra, install_order, longest_path_dag)


@dataclass
class ToolResult:
    """A tool's output.

    `summary` and `answer_text` are separate, and the separation was forced by a
    measurement rather than chosen for tidiness.

    `summary` is prose with a lead-in ("Correct install order for typer: a, b, c").
    That reads well in a planning trace. But when it is handed to an instruction-tuned
    seq2seq model that is then asked "what is the correct install order for typer?",
    the model does EXTRACTIVE QA over it and returns the shortest span that answers the
    question - which is the package name. Measured on flan-t5-base: 0.12 validity, and
    the raw output for a 5-package closure was the single word "typer".

    `answer_text` is the same information with the lead-in removed - a bare list. Same
    model, same facts, same budget: 0.75 validity, 0.95 coverage. A 6.25x difference
    that has nothing to do with the model's capability and everything to do with the
    shape of the text a tool hands it.
    """

    tool: str
    ok: bool
    summary: str
    payload: Dict[str, object]
    truncated: bool = False
    answer_text: str = ""

    def __post_init__(self) -> None:
        if not self.answer_text:
            self.answer_text = self.summary


class Toolbox:
    def __init__(self, graph: Graph, retriever=None, max_items: int = 60):
        self.graph = graph
        self.retriever = retriever
        self.max_items = max_items
        self.calls: List[str] = []

    def _truncate(self, items: List[str]) -> tuple[List[str], bool]:
        if len(items) <= self.max_items:
            return items, False
        return items[: self.max_items], True

    # -- graph tools ---------------------------------------------------------

    def direct_dependencies(self, package: str) -> ToolResult:
        self.calls.append("direct_dependencies")
        if package not in self.graph.nodes:
            return ToolResult("direct_dependencies", False,
                              f"{package} is not in the graph.", {})
        deps = sorted(self.graph.neighbours(package))
        shown, truncated = self._truncate(deps)
        summary = (f"{package} directly requires {len(deps)}: {', '.join(shown)}"
                   if deps else f"{package} has no dependencies.")
        if truncated:
            summary += f" (showing {len(shown)} of {len(deps)})"
        return ToolResult("direct_dependencies", True, summary,
                          {"dependencies": deps}, truncated)

    def full_closure(self, package: str) -> ToolResult:
        self.calls.append("full_closure")
        if package not in self.graph.nodes:
            return ToolResult("full_closure", False, f"{package} is not in the graph.", {})
        closure = sorted(self.graph.subgraph_reachable_from([package]).nodes)
        shown, truncated = self._truncate(closure)
        summary = f"Installing {package} pulls in {len(closure)} packages: {', '.join(shown)}"
        if truncated:
            summary += f" (showing {len(shown)} of {len(closure)})"
        return ToolResult("full_closure", True, summary, {"closure": closure}, truncated)

    def dependency_depth(self, package: str) -> ToolResult:
        self.calls.append("dependency_depth")
        try:
            depth, chain = longest_path_dag(self.graph, package)
        except (CycleError, ValueError) as exc:
            return ToolResult("dependency_depth", False,
                              f"{package} has no well-defined depth: {exc}", {})
        return ToolResult("dependency_depth", True,
                          f"{package} has dependency depth {depth}. Longest chain: "
                          f"{' -> '.join(chain)}",
                          {"depth": depth, "chain": chain})

    def find_cycles(self, package: Optional[str] = None) -> ToolResult:
        self.calls.append("find_cycles")
        target = (self.graph.subgraph_reachable_from([package])
                  if package and package in self.graph.nodes else self.graph)
        cycles = detect_cycles(target)
        if not cycles:
            return ToolResult("find_cycles", True, "No circular dependencies.",
                              {"cycles": []})
        summary = f"{len(cycles)} circular dependenc(ies). Shortest: " \
                  f"{' -> '.join(cycles[0])} -> {cycles[0][0]}"
        return ToolResult("find_cycles", True, summary, {"cycles": cycles})

    def shortest_dependency_path(self, source: str, target: str) -> ToolResult:
        self.calls.append("shortest_dependency_path")
        if source not in self.graph.nodes or target not in self.graph.nodes:
            return ToolResult("shortest_dependency_path", False,
                              "one or both packages are not in the graph.", {})
        dist, prev, expanded = dijkstra(self.graph, source, target)
        if dist.get(target, float("inf")) == float("inf"):
            return ToolResult("shortest_dependency_path", True,
                              f"{source} does not depend on {target}, directly or "
                              f"transitively.", {"reachable": False})
        path, cur = [], target
        while cur is not None:
            path.append(cur)
            cur = prev.get(cur)
        path.reverse()
        return ToolResult("shortest_dependency_path", True,
                          f"{source} reaches {target} in {int(dist[target])} hops: "
                          f"{' -> '.join(path)}",
                          {"path": path, "hops": int(dist[target]),
                           "nodes_expanded": expanded})

    def correct_install_order(self, package: str) -> ToolResult:
        """Ground truth from the algorithm.

        Present so the agent CAN be correct when it chooses to use it. That is
        deliberate: the evaluation is about whether the agent uses its tools well as
        the problem grows, not about whether a language model can do topological sort
        in its head. An agent that calls this and copies the answer should score 100%,
        and the finding is that past a certain depth it stops being able to.
        """
        self.calls.append("correct_install_order")
        try:
            order = install_order(self.graph, [package])
        except CycleError as exc:
            return ToolResult("correct_install_order", False,
                              f"no valid order: {exc}", {})
        shown, truncated = self._truncate(order)
        summary = f"Correct install order for {package}: {', '.join(shown)}"
        if truncated:
            summary += f" (TRUNCATED: showing {len(shown)} of {len(order)})"
        # The bare list, with no "for {package}:" framing. See ToolResult's docstring -
        # the lead-in is what triggers short-span extraction and it costs 6.25x
        # validity.
        answer_text = ", ".join(shown)
        return ToolResult("correct_install_order", True, summary,
                          {"order": order}, truncated, answer_text=answer_text)

    # -- retrieval -----------------------------------------------------------

    def search_docs(self, query: str, k: int = 4) -> ToolResult:
        self.calls.append("search_docs")
        if self.retriever is None:
            return ToolResult("search_docs", False, "retrieval is not configured.", {})
        result = self.retriever.retrieve(query, k=k)
        if not result.hits:
            return ToolResult("search_docs", True, f"No documents for {query!r}.",
                              {"hits": []})
        summary = " | ".join(h.text for h in result.hits[:k])
        return ToolResult("search_docs", True, summary,
                          {"hits": [h.as_dict() for h in result.hits]})

    # -- dispatch ------------------------------------------------------------

    def registry(self) -> Dict[str, Callable[..., ToolResult]]:
        return {
            "direct_dependencies": self.direct_dependencies,
            "full_closure": self.full_closure,
            "dependency_depth": self.dependency_depth,
            "find_cycles": self.find_cycles,
            "shortest_dependency_path": self.shortest_dependency_path,
            "correct_install_order": self.correct_install_order,
            "search_docs": self.search_docs,
        }

    def describe(self) -> str:
        return "\n".join([
            "direct_dependencies(package) - what a package directly requires",
            "full_closure(package) - every package installed transitively",
            "dependency_depth(package) - how deep the dependency tree goes",
            "find_cycles(package) - circular dependencies",
            "shortest_dependency_path(a, b) - how a reaches b",
            "correct_install_order(package) - a valid install order",
            "search_docs(query) - search package documentation",
        ])
