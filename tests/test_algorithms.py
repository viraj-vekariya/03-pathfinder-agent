"""Graph algorithms: correctness properties, not golden outputs.

A dependency graph normally admits many valid topological orders, so these assert
PROPERTIES ("no package precedes its dependency") rather than one blessed sequence.
Asserting a specific order would fail correct implementations and pass on a tie-break
change, which is exactly backwards.
"""

import pytest

from graph.algorithms import (CycleError, Graph, a_star, depth_map, detect_cycles,
                              dijkstra, install_order, longest_path_dag,
                              topological_sort)


@pytest.fixture()
def diamond():
    """app -> {web, db} -> ... -> sock. A diamond plus a shared leaf."""
    g = Graph()
    for a, b in [("app", "web"), ("app", "db"), ("web", "http"), ("web", "json"),
                 ("db", "driver"), ("driver", "http"), ("http", "sock")]:
        g.add_edge(a, b)
    return g


def _no_violations(graph: Graph, order):
    position = {n: i for i, n in enumerate(order)}
    return [(u, v) for u in graph.nodes for v in graph.neighbours(u)
            if v in position and position[v] > position[u]]


def test_install_order_puts_dependencies_first(diamond):
    """The defining property. An earlier version of Kahn's algorithm decremented
    PREDECESSORS instead of successors and reported a cycle on this acyclic graph."""
    order = install_order(diamond, ["app"])
    assert set(order) == diamond.nodes
    assert _no_violations(diamond, order) == []
    assert order[-1] == "app", "the root must be installed last"


def test_topological_sort_is_deterministic_with_a_tie_break(diamond):
    a = topological_sort(diamond, tie_break=lambda n: n)
    b = topological_sort(diamond, tie_break=lambda n: n)
    assert a == b


def test_a_cycle_is_detected_and_refuses_a_total_order(diamond):
    diamond.add_edge("sock", "app")
    cycles = detect_cycles(diamond)
    assert len(cycles) == 1
    assert set(cycles[0]) == {"app", "web", "http", "sock"} | {"db", "driver"}
    with pytest.raises(CycleError):
        topological_sort(diamond)


def test_topological_sort_and_cycle_detection_agree(diamond):
    """The consistency check that caught the Kahn bug: if the sort claims a cycle,
    the detector must find one."""
    assert detect_cycles(diamond) == []
    topological_sort(diamond)          # must not raise

    diamond.add_edge("sock", "app")
    assert detect_cycles(diamond) != []
    with pytest.raises(CycleError):
        topological_sort(diamond)


def test_a_self_loop_is_a_cycle():
    g = Graph()
    g.add_edge("a", "a")
    assert detect_cycles(g) == [["a"]]


def test_two_disjoint_cycles_are_both_found():
    g = Graph()
    for a, b in [("a", "b"), ("b", "a"), ("c", "d"), ("d", "c")]:
        g.add_edge(a, b)
    assert len(detect_cycles(g)) == 2


def test_dijkstra_finds_the_shortest_distance(diamond):
    dist, prev, expanded = dijkstra(diamond, "app")
    assert dist["sock"] == 3.0          # app -> web -> http -> sock
    assert dist["json"] == 2.0
    assert expanded == len(diamond.nodes)


def test_dijkstra_marks_unreachable_nodes_as_infinite(diamond):
    diamond.add_node("orphan")
    dist, _, _ = dijkstra(diamond, "app")
    assert dist["orphan"] == float("inf")


def test_a_star_with_a_zero_heuristic_equals_dijkstra(diamond):
    """A* IS Dijkstra when h=0. If this diverges, one of the two is wrong."""
    dist, _, _ = dijkstra(diamond, "app", "sock")
    cost, path, _ = a_star(diamond, "app", "sock", lambda a, b: 0.0)
    assert cost == dist["sock"]
    assert path[0] == "app" and path[-1] == "sock"


def test_a_star_returns_an_actual_path(diamond):
    _, path, _ = a_star(diamond, "app", "sock", lambda a, b: 0.0)
    for u, v in zip(path, path[1:]):
        assert v in diamond.neighbours(u), f"{u} -> {v} is not an edge"


def test_longest_path_is_the_depth(diamond):
    depth, chain = longest_path_dag(diamond, "app")
    assert depth == 4
    assert chain[0] == "app"
    assert len(chain) == depth + 1


def test_depth_map_is_bfs_levels(diamond):
    levels = depth_map(diamond, "app")
    assert levels["app"] == 0
    assert levels["web"] == levels["db"] == 1
    assert levels["sock"] == 3


def test_reachable_subgraph_excludes_the_unreachable(diamond):
    diamond.add_edge("unrelated", "orphan")
    sub = diamond.subgraph_reachable_from(["app"])
    assert "unrelated" not in sub.nodes and "orphan" not in sub.nodes
    assert "sock" in sub.nodes


def test_reverse_is_an_involution(diamond):
    twice = diamond.reverse().reverse()
    assert twice.nodes == diamond.nodes
    for node in diamond.nodes:
        assert sorted(twice.neighbours(node)) == sorted(diamond.neighbours(node))


def test_algorithms_survive_a_deep_chain():
    """4,000 nodes deep. Recursive implementations blow CPython's stack here, which is
    why every algorithm in this module is iterative."""
    g = Graph()
    for i in range(4000):
        g.add_edge(f"n{i}", f"n{i+1}")
    assert detect_cycles(g) == []
    order = topological_sort(g)
    assert len(order) == 4001
    depth, _ = longest_path_dag(g, "n0")
    assert depth == 4000
