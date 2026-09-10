"""Assert the browser JavaScript produces the same install orders as the Python.

The static demo at docs/ reimplements Kahn, Tarjan and the BFS depth walk in JavaScript so
it can run with no backend. That is only honest if the two implementations agree, so this
runs both over the real graph and compares them exactly.

Run:  python3 tools/check_js_matches_python.py
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from graph.build import load                              # noqa: E402
from graph.planner import build_plan                      # noqa: E402
from graph.algorithms import detect_cycles, longest_path_dag  # noqa: E402

SAMPLE = ["flask", "transformers", "langchain", "dvc", "jupyterlab", "pandas",
          "httpx", "celery", "streamlit", "poetry"]


def main() -> int:
    graph, _ = load(ROOT / "data" / "dependency_graph.json")
    py_orders = {p: build_plan(graph, p).order for p in SAMPLE}
    py_depths = {p: longest_path_dag(graph, p)[0] for p in SAMPLE}
    py_cycles = [sorted(c) for c in detect_cycles(graph)]

    script = f"""
const fs = require('fs');
const {{installOrder, depthOf, findCycles}} = require('{ROOT / "docs" / "algorithms.js"}');
const g = JSON.parse(fs.readFileSync('{ROOT / "docs" / "graph.json"}', 'utf8'));
const sample = {json.dumps(SAMPLE)};
const orders = {{}}, depths = {{}};
for(const p of sample){{ orders[p] = installOrder(g.edges, p); depths[p] = depthOf(g.edges, p).max; }}
console.log(JSON.stringify({{orders, depths, cycles: findCycles(g.edges, g.nodes)}}));
"""
    tmp = ROOT / "tools" / "_check.js"
    tmp.write_text(script)
    proc = subprocess.run(["node", str(tmp)], capture_output=True, text=True)
    tmp.unlink(missing_ok=True)
    if proc.returncode != 0:
        print(proc.stderr[:800], file=sys.stderr)
        return 1
    js = json.loads(proc.stdout)

    ok = True
    print(f"  {'package':<14} {'python':>8} {'js':>8}  {'orders match':>13} {'depth match':>12}")
    for p in SAMPLE:
        same_order = js["orders"][p] == py_orders[p]
        same_depth = js["depths"][p] == py_depths[p]
        ok &= same_order and same_depth
        print(f"  {p:<14} {len(py_orders[p]):>8} {len(js['orders'][p] or []):>8}  "
              f"{str(same_order):>13} {str(same_depth):>12}")

    js_cycles = [sorted(c) for c in js["cycles"]]
    same_cycles = js_cycles == py_cycles
    ok &= same_cycles
    print(f"\n  cycles: python {len(py_cycles)}, js {len(js_cycles)}, identical: {same_cycles}")
    print("\n  JavaScript agrees with Python exactly" if ok
          else "\n  *** THE TWO IMPLEMENTATIONS DIVERGE ***", file=sys.stderr if not ok else sys.stdout)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
