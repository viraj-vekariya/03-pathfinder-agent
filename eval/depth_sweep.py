"""THE MEASUREMENT: how does plan quality change as dependency depth grows?

The agent has a tool that returns the exact answer. So in principle accuracy should be
flat at 100% regardless of depth - the task never gets harder, only bigger. This sweep
asks whether it actually is.

Design decisions that make the result mean something:

* **Real packages at every depth.** Tasks are drawn from the real PyPI graph, grouped by
  their measured dependency depth. Nothing is synthesised to produce a curve.
* **Sampled within depth, seeded.** Depth 1 has 89 candidate packages and depth 9 has 2;
  running all of them would weight the average toward shallow tasks. A fixed sample per
  depth, drawn with a fixed seed, keeps each depth equally represented and the whole
  sweep reproducible.
* **Two metrics, not one.** `valid` is all-or-nothing and hides partial progress;
  `coverage` shows how much of the closure survived. A curve where validity collapses
  while coverage stays high means something different from one where both fall.
* **Every prompt and completion is measured.** Token counts, truncation flags and
  dropped-token counts are recorded per run, so eval/truncation.py can attribute the
  failure rather than speculate about it.

Run:  python3 -m eval.depth_sweep [--per-depth 6] [--backend flan-t5]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.llm import get_llm                                    # noqa: E402
from agent.loop import PathfinderAgent                           # noqa: E402
from agent.tools import Toolbox                                  # noqa: E402
from graph.build import load                                     # noqa: E402
from graph.planner import build_plan, packages_by_depth          # noqa: E402

DEFAULT_GRAPH = ROOT / "data" / "dependency_graph.json"
OUTPUTS = ROOT / "outputs"


@dataclass
class TaskResult:
    package: str
    depth: int
    closure_size: int
    valid: bool
    coverage: float
    order_violations: int
    missing: int
    reason: str
    tools_used: List[str]
    input_truncated: bool
    tokens_dropped: int
    max_prompt_tokens: int
    context_limit: int
    proposed_length: int
    seconds: float


def run_sweep(graph_path: Path = DEFAULT_GRAPH, per_depth: int = 6,
              backend: str = "auto", seed: int = 20260910,
              max_depth: Optional[int] = None) -> Dict[str, object]:
    graph, metadata = load(graph_path)
    llm = get_llm(backend)
    agent = PathfinderAgent(graph, llm, Toolbox(graph))

    buckets = packages_by_depth(graph, min_closure=2)
    rng = random.Random(seed)

    results: List[TaskResult] = []
    started = time.time()

    for depth in sorted(buckets):
        if max_depth is not None and depth > max_depth:
            continue
        candidates = buckets[depth]
        # Sample without replacement, seeded. Sorting first makes the sample independent
        # of dict iteration order, so two runs on the same graph pick the same tasks.
        chosen = rng.sample(sorted(candidates), min(per_depth, len(candidates)))

        for package, closure_size in chosen:
            run = agent.run(package)
            results.append(TaskResult(
                package=package,
                depth=depth,
                closure_size=closure_size,
                valid=run.grade.valid,
                coverage=run.grade.coverage,
                order_violations=run.grade.order_violations,
                missing=len(run.grade.missing),
                reason=run.grade.reason[:160],
                tools_used=run.tools_used,
                input_truncated=run.any_truncation,
                tokens_dropped=run.total_tokens_dropped,
                max_prompt_tokens=run.max_prompt_tokens,
                context_limit=run.context_limit,
                proposed_length=len(run.proposed),
                seconds=run.seconds,
            ))
            print(f"  depth {depth}  {package:<32} closure {closure_size:>3}  "
                  f"valid={str(run.grade.valid):<5} cov={run.grade.coverage:.2f}  "
                  f"tok={run.max_prompt_tokens:>4}"
                  f"{'  TRUNCATED' if run.any_truncation else ''}", flush=True)

    by_depth: Dict[int, Dict[str, object]] = {}
    for depth in sorted({r.depth for r in results}):
        rows = [r for r in results if r.depth == depth]
        by_depth[depth] = {
            "n": len(rows),
            "validity": round(sum(r.valid for r in rows) / len(rows), 4),
            "mean_coverage": round(statistics.fmean(r.coverage for r in rows), 4),
            "mean_closure_size": round(statistics.fmean(r.closure_size for r in rows), 1),
            "mean_prompt_tokens": round(statistics.fmean(r.max_prompt_tokens for r in rows), 1),
            "truncation_rate": round(sum(r.input_truncated for r in rows) / len(rows), 4),
            "mean_tokens_dropped": round(statistics.fmean(r.tokens_dropped for r in rows), 1),
            "mean_order_violations": round(statistics.fmean(r.order_violations for r in rows), 2),
        }

    valid_rows = [r for r in results if r.valid]
    failed_rows = [r for r in results if not r.valid]

    # Where does it break? The first depth at which fewer than half the tasks succeed.
    collapse_depth = next(
        (d for d in sorted(by_depth) if by_depth[d]["validity"] < 0.5), None)

    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "backend": llm.backend,
        "model": llm.model,
        "context_limit": llm.context_limit,
        "graph": {"nodes": len(graph.nodes),
                  "edges": sum(len(v) for v in graph.edges.values())},
        "config": {"per_depth": per_depth, "seed": seed},
        "tasks": len(results),
        "overall_validity": round(sum(r.valid for r in results) / len(results), 4),
        "overall_coverage": round(statistics.fmean(r.coverage for r in results), 4),
        "by_depth": {str(k): v for k, v in by_depth.items()},
        "collapse_depth": collapse_depth,
        "deepest_success": max((r.depth for r in valid_rows), default=None),
        "shallowest_failure": min((r.depth for r in failed_rows), default=None),
        "seconds": round(time.time() - started, 1),
        "results": [asdict(r) for r in results],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-depth", type=int, default=6)
    ap.add_argument("--backend", default="flan-t5")
    ap.add_argument("--seed", type=int, default=20260910)
    ap.add_argument("--max-depth", type=int, default=None)
    ap.add_argument("--graph", default=str(DEFAULT_GRAPH))
    args = ap.parse_args()

    print(f"depth sweep: {args.per_depth} packages per depth, backend={args.backend}\n")
    report = run_sweep(Path(args.graph), args.per_depth, args.backend,
                       args.seed, args.max_depth)

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "depth_sweep.json").write_text(json.dumps(report, indent=2) + "\n")

    print(f"\n  {'depth':>5} {'n':>3} {'validity':>9} {'coverage':>9} {'closure':>8} "
          f"{'tokens':>7} {'trunc%':>7}")
    for depth, row in report["by_depth"].items():
        print(f"  {depth:>5} {row['n']:>3} {row['validity']:>9.2f} "
              f"{row['mean_coverage']:>9.2f} {row['mean_closure_size']:>8.1f} "
              f"{row['mean_prompt_tokens']:>7.0f} {row['truncation_rate'] * 100:>6.0f}%")

    print(f"\n  overall validity  {report['overall_validity']:.3f}")
    print(f"  overall coverage  {report['overall_coverage']:.3f}")
    print(f"  collapse depth    {report['collapse_depth']} "
          f"(first depth where fewer than half the plans are valid)")
    print(f"  deepest success   {report['deepest_success']}")
    print(f"\n  wrote outputs/depth_sweep.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
