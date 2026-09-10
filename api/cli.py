"""Command line for the same operations the API exposes.

The CLI calls the same functions the API does rather than reimplementing them. Two
surfaces over one implementation means a fix lands in both; two implementations means
they drift and one of them is quietly wrong.

    python3 -m api.cli plan flask
    python3 -m api.cli path jupyterlab typing-extensions
    python3 -m api.cli cycles
    python3 -m api.cli search "what does celery need"
    python3 -m api.cli agent flask
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from graph.algorithms import a_star, detect_cycles, dijkstra      # noqa: E402
from graph.build import load                                      # noqa: E402
from graph.heuristics import (compare_expansions, make_depth_heuristic,   # noqa: E402
                              make_out_degree_heuristic, zero_heuristic)
from graph.planner import build_plan                              # noqa: E402

GRAPH_PATH = ROOT / "data" / "dependency_graph.json"


def _graph():
    if not GRAPH_PATH.exists():
        print(f"no graph at {GRAPH_PATH}; run: python3 -m graph.build", file=sys.stderr)
        raise SystemExit(1)
    return load(GRAPH_PATH)


def cmd_plan(args) -> int:
    graph, _ = _graph()
    result = build_plan(graph, args.package)
    if args.json:
        print(json.dumps(result.as_dict(), indent=2))
        return 0
    print(f"{args.package}: depth {result.depth}, {result.closure_size} packages")
    if result.cycles:
        print(f"  WARNING: {len(result.cycles)} cycle(s) - no total order exists")
        for cycle in result.cycles[:2]:
            print(f"    {' -> '.join(cycle)} -> {cycle[0]}")
        return 2
    for i, package in enumerate(result.order, 1):
        print(f"  {i:>3}. {package}")
    return 0


def cmd_path(args) -> int:
    graph, _ = _graph()
    heuristics = {"zero": zero_heuristic,
                  "depth": make_depth_heuristic(graph, args.source),
                  "out_degree": make_out_degree_heuristic(graph)}
    comparison = compare_expansions(graph, args.source, args.target, heuristics)
    _, _, dij = dijkstra(graph, args.source, args.target)

    print(f"{args.source} -> {args.target}")
    for name, row in comparison.items():
        status = "admissible" if row["admissible"] else "NOT ADMISSIBLE"
        print(f"  A* [{name:<11}] cost {row['cost']}  expanded {row['nodes_expanded']:>4}"
              f"  ({status})")
    print(f"  Dijkstra        expanded {dij:>4}")
    agree = all(r["agrees_on_optimal_cost"] for r in comparison.values())
    print(f"  all heuristics agree on the optimal cost: {agree}")
    return 0


def cmd_cycles(args) -> int:
    graph, _ = _graph()
    found = detect_cycles(graph)
    print(f"{len(found)} circular dependenc(ies) in {len(graph.nodes)} packages")
    for cycle in found:
        print(f"  {' -> '.join(cycle)} -> {cycle[0]}")
    return 0


def cmd_search(args) -> int:
    graph, metadata = _graph()
    from rag.corpus import build_corpus
    from rag.embed import get_embedder
    from rag.retrieve import Retriever
    from rag.store import NumpyStore

    docs = build_corpus(graph, metadata)
    embedder = get_embedder("auto")
    vectors = embedder.encode([d.text for d in docs])
    store = NumpyStore(embedder.dim)
    store.add([d.doc_id for d in docs], [d.package for d in docs],
              [d.kind for d in docs], [d.text for d in docs],
              [d.metadata for d in docs], vectors)
    result = Retriever(store, embedder, sorted(graph.nodes)).retrieve(args.query, k=args.k)
    print(f"query: {args.query}  (kind={result.inferred_kind}, "
          f"mentioned={result.mentioned})")
    for hit in result.hits:
        print(f"  {hit.score:.3f} [{hit.kind}] {hit.text[:120]}")
    return 0


def cmd_agent(args) -> int:
    graph, metadata = _graph()
    from agent.llm import get_llm
    from agent.loop import PathfinderAgent
    from agent.tools import Toolbox

    run = PathfinderAgent(graph, get_llm("auto"), Toolbox(graph)).run(args.package)
    print(json.dumps(run.as_dict(), indent=2))
    print("\nproposed order:")
    print("  " + ", ".join(run.proposed))
    return 0 if run.grade.valid else 1


def main() -> int:
    ap = argparse.ArgumentParser(prog="pathfinder")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("plan"); p.add_argument("package"); p.add_argument("--json", action="store_true"); p.set_defaults(fn=cmd_plan)
    p = sub.add_parser("path"); p.add_argument("source"); p.add_argument("target"); p.set_defaults(fn=cmd_path)
    p = sub.add_parser("cycles"); p.set_defaults(fn=cmd_cycles)
    p = sub.add_parser("search"); p.add_argument("query"); p.add_argument("-k", type=int, default=5); p.set_defaults(fn=cmd_search)
    p = sub.add_parser("agent"); p.add_argument("package"); p.set_defaults(fn=cmd_agent)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
