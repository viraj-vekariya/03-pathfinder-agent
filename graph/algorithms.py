"""Graph algorithms, written out rather than imported.

networkx would do all of this in one line each. It is deliberately not used, for one
reason: this file is the part of the project that has to be defensible line by line in
an interview, and "I called nx.topological_sort" is not an answer to "what happens when
the graph has a cycle?"

Everything here is iterative. Real package dependency graphs reach depths that overflow
CPython's default 1000-frame recursion limit, and a resolver that crashes on a deep
graph is a resolver that fails exactly when the problem is hardest.

Complexities, for the graphs this actually runs on (V packages, E dependency edges):

    topological_sort      O(V + E)      Kahn, iterative
    detect_cycles         O(V + E)      iterative Tarjan SCC
    dijkstra              O(E log V)    binary heap
    a_star                O(E log V)    same bound, fewer expansions in practice
    longest_path_dag      O(V + E)      DP over a topological order
"""

from __future__ import annotations

import heapq
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

INF = float("inf")


class CycleError(ValueError):
    """Raised when a total order is requested from a graph that has no total order."""

    def __init__(self, cycles: List[List[str]]):
        self.cycles = cycles
        preview = " -> ".join(cycles[0] + [cycles[0][0]]) if cycles else "?"
        super().__init__(f"{len(cycles)} cycle(s); shortest: {preview}")


@dataclass
class Graph:
    """Directed graph. `edges[u]` lists what u depends on.

    Adjacency dict rather than a matrix: dependency graphs are extremely sparse
    (a package has ~3 dependencies out of ~500 nodes), so a matrix would be 99.4%
    zeroes and turn O(V+E) traversals into O(V^2).
    """

    edges: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    weights: Dict[Tuple[str, str], float] = field(default_factory=dict)
    nodes: Set[str] = field(default_factory=set)

    def add_node(self, node: str) -> None:
        self.nodes.add(node)
        self.edges.setdefault(node, [])

    def add_edge(self, src: str, dst: str, weight: float = 1.0) -> None:
        self.add_node(src)
        self.add_node(dst)
        if dst not in self.edges[src]:
            self.edges[src].append(dst)
        self.weights[(src, dst)] = weight

    def neighbours(self, node: str) -> List[str]:
        return self.edges.get(node, [])

    def weight(self, src: str, dst: str) -> float:
        return self.weights.get((src, dst), 1.0)

    def reverse(self) -> "Graph":
        r = Graph()
        for node in self.nodes:
            r.add_node(node)
        for src, dsts in self.edges.items():
            for dst in dsts:
                r.add_edge(dst, src, self.weight(src, dst))
        return r

    def in_degrees(self) -> Dict[str, int]:
        deg = {n: 0 for n in self.nodes}
        for src, dsts in self.edges.items():
            for dst in dsts:
                deg[dst] = deg.get(dst, 0) + 1
        return deg

    def subgraph_reachable_from(self, roots: Iterable[str]) -> "Graph":
        """BFS-induced subgraph. Used to isolate one package's dependency closure."""
        sub = Graph()
        seen, queue = set(), deque(roots)
        while queue:
            node = queue.popleft()
            if node in seen:
                continue
            seen.add(node)
            sub.add_node(node)
            for nxt in self.neighbours(node):
                sub.add_edge(node, nxt, self.weight(node, nxt))
                if nxt not in seen:
                    queue.append(nxt)
        return sub

    def stats(self) -> Dict[str, object]:
        edge_count = sum(len(v) for v in self.edges.values())
        return {"nodes": len(self.nodes), "edges": edge_count,
                "density": round(edge_count / max(1, len(self.nodes) ** 2), 6),
                "mean_out_degree": round(edge_count / max(1, len(self.nodes)), 2)}


# ---------------------------------------------------------------------------
# Topological sort
# ---------------------------------------------------------------------------

def topological_sort(graph: Graph, tie_break: Optional[Callable[[str], object]] = None
                     ) -> List[str]:
    """Kahn's algorithm. Returns dependencies BEFORE the things that need them.

    Kahn rather than DFS post-order for two reasons that matter here:
      * it detects a cycle naturally - if the queue empties with nodes left over, the
        remainder is exactly the cyclic part, which is what the caller needs to be told;
      * it is iterative, so a 4,000-deep chain does not blow the Python stack.

    `tie_break` makes the output deterministic when several nodes are simultaneously
    ready. Without it the order depends on dict insertion order, and a resolver that
    produces a different valid answer each run is impossible to test or to diff.
    """
    in_degree = graph.in_degrees()

    ready = [n for n in graph.nodes if in_degree[n] == 0]
    if tie_break:
        ready.sort(key=tie_break)
    queue = deque(ready)

    order: List[str] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        newly_ready = []
        # Removing `node` removes its OUTGOING edges, so it is the in-degree of node's
        # SUCCESSORS that falls. An earlier version walked the reversed graph here and
        # decremented predecessors instead, which left every real successor stuck above
        # zero - the queue drained early and the function reported a cycle on a graph
        # that provably had none (detect_cycles returned an empty list at the same
        # moment). A disagreement between two algorithms on the same graph is the
        # cheapest bug signal there is.
        for successor in graph.neighbours(node):
            in_degree[successor] -= 1
            if in_degree[successor] == 0:
                newly_ready.append(successor)
        if tie_break:
            newly_ready.sort(key=tie_break)
        queue.extend(newly_ready)

    if len(order) != len(graph.nodes):
        raise CycleError(detect_cycles(graph))
    return order


def install_order(graph: Graph, roots: Iterable[str]) -> List[str]:
    """Topological order of one package's closure, dependencies first.

    Note the reverse: `edges[u]` points at what u DEPENDS ON, so a plain topological
    sort of that graph yields dependents before dependencies - the exact opposite of an
    install order. Reversing first is the whole subtlety, and getting it backwards
    produces an order that looks plausible and fails on the first install.
    """
    closure = graph.subgraph_reachable_from(roots)
    return topological_sort(closure.reverse(), tie_break=lambda n: n)


# ---------------------------------------------------------------------------
# Cycle detection - iterative Tarjan
# ---------------------------------------------------------------------------

def detect_cycles(graph: Graph) -> List[List[str]]:
    """Every strongly connected component with more than one node, plus self-loops.

    Tarjan rather than "DFS and remember the stack": Tarjan finds ALL SCCs in one pass,
    which is what you want when a lockfile is broken in several places at once and you
    would rather report every cycle than the first one.

    Written iteratively with an explicit frame stack. Recursive Tarjan is shorter and
    dies on real graphs.
    """
    index_counter = 0
    index: Dict[str, int] = {}
    lowlink: Dict[str, int] = {}
    on_stack: Dict[str, bool] = {}
    stack: List[str] = []
    components: List[List[str]] = []

    for root in sorted(graph.nodes):
        if root in index:
            continue
        # Each frame is (node, iterator over its neighbours).
        work: List[Tuple[str, Iterable[str]]] = [(root, iter(graph.neighbours(root)))]
        index[root] = lowlink[root] = index_counter
        index_counter += 1
        stack.append(root)
        on_stack[root] = True

        while work:
            node, iterator = work[-1]
            advanced = False
            for nxt in iterator:
                if nxt not in index:
                    index[nxt] = lowlink[nxt] = index_counter
                    index_counter += 1
                    stack.append(nxt)
                    on_stack[nxt] = True
                    work.append((nxt, iter(graph.neighbours(nxt))))
                    advanced = True
                    break
                if on_stack.get(nxt):
                    lowlink[node] = min(lowlink[node], index[nxt])
            if advanced:
                continue

            work.pop()
            if work:
                parent = work[-1][0]
                lowlink[parent] = min(lowlink[parent], lowlink[node])

            if lowlink[node] == index[node]:
                component = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    component.append(w)
                    if w == node:
                        break
                # A single node is only a cycle if it points at itself.
                if len(component) > 1 or node in graph.neighbours(node):
                    components.append(sorted(component))

    components.sort(key=len)
    return components


# ---------------------------------------------------------------------------
# Shortest paths
# ---------------------------------------------------------------------------

def dijkstra(graph: Graph, source: str, target: Optional[str] = None
             ) -> Tuple[Dict[str, float], Dict[str, Optional[str]], int]:
    """Lazy-deletion Dijkstra. Returns (distances, predecessors, nodes_expanded).

    `nodes_expanded` is returned because it is the only fair way to compare this with
    A*: wall-clock on a graph this small is dominated by noise, while expansions are a
    deterministic count of the work each algorithm actually did.

    Lazy deletion (push duplicates, skip stale pops) rather than a decrease-key heap:
    Python's heapq has no decrease-key, and the alternatives - an indexed heap, or
    marking entries dead - cost more code and more bugs than the duplicate entries cost
    memory on a sparse graph.

    Requires non-negative weights. Every weight here is a package's install cost, which
    cannot be negative; if that ever changed, this would silently return wrong answers
    and Bellman-Ford would be the right call instead.
    """
    dist: Dict[str, float] = {n: INF for n in graph.nodes}
    prev: Dict[str, Optional[str]] = {n: None for n in graph.nodes}
    dist[source] = 0.0
    visited: Set[str] = set()
    expanded = 0

    heap: List[Tuple[float, str]] = [(0.0, source)]
    while heap:
        d, node = heapq.heappop(heap)
        if node in visited:
            continue          # a stale duplicate; the real entry was popped earlier
        visited.add(node)
        expanded += 1
        if target is not None and node == target:
            break
        for nxt in graph.neighbours(node):
            nd = d + graph.weight(node, nxt)
            if nd < dist.get(nxt, INF):
                dist[nxt] = nd
                prev[nxt] = node
                heapq.heappush(heap, (nd, nxt))
    return dist, prev, expanded


def a_star(graph: Graph, source: str, target: str,
           heuristic: Callable[[str, str], float]
           ) -> Tuple[float, List[str], int]:
    """A* with the same lazy-deletion heap. Returns (cost, path, nodes_expanded).

    A* is Dijkstra with the priority changed from g to g + h. It is only guaranteed to
    return an optimal path if the heuristic is ADMISSIBLE - never over-estimating the
    true remaining cost. graph/heuristics.py explains why the one used here is, and
    includes a test that would catch it becoming inadmissible.
    """
    g_score: Dict[str, float] = {source: 0.0}
    prev: Dict[str, Optional[str]] = {source: None}
    visited: Set[str] = set()
    expanded = 0

    heap: List[Tuple[float, float, str]] = [(heuristic(source, target), 0.0, source)]
    while heap:
        _, g, node = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        expanded += 1

        if node == target:
            path, cur = [], node
            while cur is not None:
                path.append(cur)
                cur = prev[cur]
            return g, path[::-1], expanded

        for nxt in graph.neighbours(node):
            tentative = g + graph.weight(node, nxt)
            if tentative < g_score.get(nxt, INF):
                g_score[nxt] = tentative
                prev[nxt] = node
                heapq.heappush(heap, (tentative + heuristic(nxt, target), tentative, nxt))

    return INF, [], expanded


def reconstruct_path(prev: Dict[str, Optional[str]], target: str) -> List[str]:
    path, cur = [], target
    while cur is not None:
        path.append(cur)
        cur = prev.get(cur)
    return path[::-1]


# ---------------------------------------------------------------------------
# Depth
# ---------------------------------------------------------------------------

def longest_path_dag(graph: Graph, source: str) -> Tuple[int, List[str]]:
    """Longest dependency chain from `source`. This is the graph's DEPTH.

    It is the independent variable of the whole evaluation: eval/depth_sweep.py asks how
    agent plan quality changes as this grows.

    Longest path is NP-hard in general but linear on a DAG, by relaxing edges in
    topological order. That is exactly why the cycle check has to come first - on a
    cyclic graph "longest path" is unbounded and the DP would be meaningless.
    """
    closure = graph.subgraph_reachable_from([source])
    order = topological_sort(closure)

    depth = {n: 0 for n in closure.nodes}
    parent: Dict[str, Optional[str]] = {n: None for n in closure.nodes}
    for node in order:
        for nxt in closure.neighbours(node):
            if depth[node] + 1 > depth[nxt]:
                depth[nxt] = depth[node] + 1
                parent[nxt] = node

    deepest = max(depth, key=lambda n: depth[n])
    chain, cur = [], deepest
    while cur is not None:
        chain.append(cur)
        cur = parent[cur]
    return depth[deepest], chain[::-1]


def depth_map(graph: Graph, source: str) -> Dict[str, int]:
    """BFS level of every node in a closure - how many hops from the root."""
    levels = {source: 0}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        for nxt in graph.neighbours(node):
            if nxt not in levels:
                levels[nxt] = levels[node] + 1
                queue.append(nxt)
    return levels
