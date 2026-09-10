"""Heuristics for A*, and the argument that they are admissible.

A* only returns an optimal path if its heuristic never OVER-estimates the true
remaining cost. That property is called admissibility, and it is not something to
assume - an inadmissible heuristic gives you a fast search that quietly returns wrong
answers, which is worse than a slow search.

Each heuristic below states its argument. `verify_admissible` checks the claim against
ground truth computed by Dijkstra, and the test suite runs it on the real graph, so an
edit that breaks admissibility fails the build instead of silently degrading results.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

from .algorithms import Graph, depth_map, dijkstra


def zero_heuristic(_a: str, _b: str) -> float:
    """h = 0. A* degenerates to Dijkstra.

    Trivially admissible (0 never over-estimates a non-negative cost), and it is the
    control arm: any other heuristic has to beat this on node expansions to justify
    existing at all.
    """
    return 0.0


def make_depth_heuristic(graph: Graph, source: str) -> Callable[[str, str], float]:
    """h(n, goal) = max(0, level(goal) - level(n)), using BFS levels from `source`.

    ADMISSIBILITY ARGUMENT. Every edge costs >= 1, so a path from `source` through n to
    the goal has length level(n) + d(n, goal). Since BFS gives the SHORTEST distance
    from source, level(goal) <= level(n) + d(n, goal), and rearranging gives
    d(n, goal) >= level(goal) - level(n). The heuristic returns exactly that lower
    bound, clamped at zero. It never over-estimates. Admissible.

    Also CONSISTENT: across any edge n -> m, levels differ by at most 1 while w >= 1,
    so h(n) <= w(n,m) + h(m). Consistency is what lets A* skip re-expanding closed
    nodes safely.

    THE VERSION THIS REPLACED WAS WRONG, and verify_admissible() caught it rather than
    a reviewer. It returned |level(goal) - level(n)| - the ABSOLUTE difference - with a
    confident proof attached. The proof only covers the case where the goal is DEEPER
    than n. When the goal is shallower, say level(n)=5 and level(goal)=1, the absolute
    difference claims 4 hops remain while a single edge n -> goal may cost 1. That is a
    4x over-estimate, and an inadmissible heuristic makes A* return non-optimal paths
    silently - it does not error, it just quietly stops being A*.

    The lesson is the reason verify_admissible exists at all: an admissibility argument
    written in a docstring is a claim, and claims about search correctness are cheap to
    check against ground truth and expensive to get wrong.
    """
    levels = depth_map(graph, source)

    def h(node: str, goal: str) -> float:
        if node not in levels or goal not in levels:
            return 0.0          # unknown: fall back to the safe under-estimate
        return float(max(0, levels[goal] - levels[node]))

    return h


def make_out_degree_heuristic(graph: Graph) -> Callable[[str, str], float]:
    """h(n, goal) = 1 if n != goal else 0.

    ADMISSIBILITY ARGUMENT. If n is not the goal, at least one more edge must be
    traversed, and every edge costs >= 1. So the true remaining cost is >= 1.
    Admissible, and consistent for the same reason.

    Deliberately the weakest non-trivial heuristic. It is here as the honest middle
    point between "no information" and the depth heuristic, so the expansion comparison
    has three points rather than two.
    """

    def h(node: str, goal: str) -> float:
        return 0.0 if node == goal else 1.0

    return h


def verify_admissible(graph: Graph, heuristic: Callable[[str, str], float],
                      goal: str, sample: int | None = None) -> Tuple[bool, List[Dict]]:
    """Check h(n, goal) <= true_cost(n -> goal) for every reachable n.

    Ground truth comes from Dijkstra on the REVERSED graph run once from the goal, which
    gives the true cost from every node to the goal in a single O(E log V) pass rather
    than one search per node.

    Returns (ok, violations). Violations carry the node and both numbers, because
    "the heuristic is inadmissible" is not actionable and "h(urllib3)=3 but the true
    cost is 2" is.
    """
    true_cost, _, _ = dijkstra(graph.reverse(), goal)

    violations: List[Dict] = []
    nodes = sorted(n for n in graph.nodes if true_cost.get(n, float("inf")) < float("inf"))
    if sample:
        nodes = nodes[:sample]

    for node in nodes:
        estimate = heuristic(node, goal)
        actual = true_cost[node]
        # Float tolerance: these are sums of floats and an exact > would flag noise.
        if estimate > actual + 1e-9:
            violations.append({"node": node, "heuristic": estimate, "true_cost": actual,
                               "overestimate_by": round(estimate - actual, 6)})
    return len(violations) == 0, violations


def compare_expansions(graph: Graph, source: str, target: str,
                       heuristics: Dict[str, Callable[[str, str], float]]
                       ) -> Dict[str, Dict[str, object]]:
    """Run A* with each heuristic and report cost and node expansions.

    Expansions rather than wall-clock: on a graph of a few hundred nodes the timings are
    dominated by interpreter noise, while expansions are a deterministic count of the
    work each search actually did. Every admissible heuristic must return the SAME cost;
    if one does not, it is not admissible and the comparison is meaningless.
    """
    from .algorithms import a_star

    results: Dict[str, Dict[str, object]] = {}
    for name, h in heuristics.items():
        cost, path, expanded = a_star(graph, source, target, h)
        ok, violations = verify_admissible(graph, h, target)
        results[name] = {
            "cost": cost if cost != float("inf") else None,
            "path_length": len(path),
            "nodes_expanded": expanded,
            "admissible": ok,
            "violations": violations[:3],
        }

    costs = {r["cost"] for r in results.values() if r["cost"] is not None}
    for r in results.values():
        r["agrees_on_optimal_cost"] = len(costs) <= 1
    return results
