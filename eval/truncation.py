"""Attribute the depth collapse to a mechanism, instead of asserting one.

The sweep establishes THAT plan quality falls with depth. "LLMs get unreliable on long
problems" is not an explanation - it is a restatement. This file tests three candidate
mechanisms against each other, and each is falsifiable.

  M1  INPUT TRUNCATION. The prompt outgrows the model's context window and the
      tokenizer silently drops the tail, so the agent never receives packages it is
      being asked to order.
      Falsifiable by: giving the SAME tasks a larger context and seeing failures
      disappear. If they persist, M1 is not the cause.

  M2  OUTPUT TOKEN LIMIT. The answer is capped at max_new_tokens. A 90-package closure
      needs more output tokens than the cap allows, so the list is cut off mid-way -
      which looks identical to "the model forgot packages" but is not.
      Falsifiable by: comparing the required output length against the cap, and by
      raising the cap.

  M3  NEITHER. The model receives everything and is allowed to emit everything, and
      still gets it wrong - genuine reasoning failure.

These are not mutually exclusive, and separating them matters: M1 is fixed with a bigger
context, M2 with one config line, and M3 with a better model. Prescribing the wrong one
wastes the effort.

Run:  python3 -m eval.truncation
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.llm import FlanT5LLM, get_llm                        # noqa: E402
from agent.loop import PathfinderAgent                          # noqa: E402
from agent.tools import Toolbox                                 # noqa: E402
from graph.build import load                                    # noqa: E402
from graph.planner import build_plan, required_closure          # noqa: E402

OUTPUTS = ROOT / "outputs"
DEFAULT_GRAPH = ROOT / "data" / "dependency_graph.json"


def estimate_answer_tokens(llm, packages: List[str]) -> int:
    """How many output tokens the correct answer actually needs.

    Measured with the real tokenizer on the real answer string, not estimated from a
    character count - package names like `apache-airflow-providers-common-sql` tokenise
    into many more pieces than their length suggests, and an estimate would understate
    the requirement exactly where it matters.
    """
    return llm.count_tokens(", ".join(packages))


def classify_failures(sweep: Dict, llm, graph) -> Dict[str, object]:
    """Attribute each failed task to M1, M2 or M3."""
    rows = sweep["results"]
    failures = [r for r in rows if not r["valid"]]

    buckets = {"M1_input_truncation": [], "M2_output_limit": [],
               "M3_neither": [], "M1_and_M2": []}

    max_new_tokens = 320          # the cap used by the agent's answer step

    for row in failures:
        required = sorted(required_closure(graph, row["package"]))
        needed_output = estimate_answer_tokens(llm, required)
        input_truncated = row["input_truncated"]
        output_capped = needed_output > max_new_tokens

        if input_truncated and output_capped:
            key = "M1_and_M2"
        elif input_truncated:
            key = "M1_input_truncation"
        elif output_capped:
            key = "M2_output_limit"
        else:
            key = "M3_neither"

        buckets[key].append({
            "package": row["package"],
            "depth": row["depth"],
            "closure_size": row["closure_size"],
            "coverage": row["coverage"],
            "prompt_tokens": row["max_prompt_tokens"],
            "tokens_dropped": row["tokens_dropped"],
            "output_tokens_needed": needed_output,
            "output_cap": max_new_tokens,
        })

    total = len(failures) or 1
    return {
        "total_failures": len(failures),
        "attribution": {k: {"count": len(v), "share": round(len(v) / total, 4)}
                        for k, v in buckets.items()},
        "examples": {k: v[:4] for k, v in buckets.items()},
        "output_cap": max_new_tokens,
    }


def context_intervention(sweep: Dict, graph, packages: List[str],
                         big_context: int = 4096) -> Dict[str, object]:
    """M1's falsification test: rerun failing tasks with a LARGER context.

    flan-t5's 512 limit comes from its positional encoding, so it cannot simply be
    raised on the same model. Instead the harness measures what WOULD fit: for each
    failed task it recomputes the prompt and asks whether a 4k context would have
    contained it. That distinguishes "the information did not fit" from "the
    information fit and was still not used", which is exactly the M1/M3 boundary.
    """
    llm = get_llm("flan-t5")
    agent = PathfinderAgent(graph, llm, Toolbox(graph))

    rows = []
    for package in packages:
        # Rebuild the prompt the agent would send, and measure it.
        toolbox = Toolbox(graph)
        result = toolbox.correct_install_order(package)
        from agent.loop import ANSWER_PROMPT
        prompt = ANSWER_PROMPT.format(package=package, memory=result.summary)
        tokens = llm.count_tokens(prompt)
        rows.append({
            "package": package,
            "answer_prompt_tokens": tokens,
            "fits_in_512": tokens <= 512,
            "fits_in_4096": tokens <= big_context,
            "tool_output_truncated": result.truncated,
        })

    return {
        "big_context": big_context,
        "tasks": len(rows),
        "fit_in_512": sum(r["fits_in_512"] for r in rows),
        "fit_in_4096": sum(r["fits_in_4096"] for r in rows),
        "would_be_rescued_by_bigger_context": sum(
            (not r["fits_in_512"]) and r["fits_in_4096"] for r in rows),
        "rows": rows[:20],
    }


def correlations(sweep: Dict) -> Dict[str, object]:
    """Correlate validity against each candidate driver.

    Pearson r on a binary outcome is a point-biserial correlation, which is legitimate
    and interpretable here. The point is comparative: whichever driver correlates most
    strongly with failure is the one worth fixing first. Reported alongside the raw
    group means, because a correlation without the means hides effect size.
    """
    rows = sweep["results"]

    def r(xs: List[float], ys: List[float]) -> Optional[float]:
        n = len(xs)
        if n < 3:
            return None
        mx, my = statistics.fmean(xs), statistics.fmean(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        dx = sum((x - mx) ** 2 for x in xs) ** 0.5
        dy = sum((y - my) ** 2 for y in ys) ** 0.5
        return round(num / (dx * dy), 4) if dx and dy else None

    valid = [1.0 if x["valid"] else 0.0 for x in rows]
    return {
        "validity_vs_depth": r([x["depth"] for x in rows], valid),
        "validity_vs_closure_size": r([x["closure_size"] for x in rows], valid),
        "validity_vs_prompt_tokens": r([x["max_prompt_tokens"] for x in rows], valid),
        "validity_vs_tokens_dropped": r([x["tokens_dropped"] for x in rows], valid),
        "coverage_vs_closure_size": r([x["closure_size"] for x in rows],
                                      [x["coverage"] for x in rows]),
        "note": ("Negative r means the driver predicts FAILURE. The strongest negative "
                 "correlate is the mechanism worth fixing first."),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default=str(OUTPUTS / "depth_sweep.json"))
    ap.add_argument("--graph", default=str(DEFAULT_GRAPH))
    args = ap.parse_args()

    sweep_path = Path(args.sweep)
    if not sweep_path.exists():
        print("run `python3 -m eval.depth_sweep` first", file=sys.stderr)
        return 1

    sweep = json.loads(sweep_path.read_text())
    graph, _ = load(Path(args.graph))
    llm = get_llm("flan-t5")

    print("attributing failures to a mechanism\n")
    attribution = classify_failures(sweep, llm, graph)
    for name, stats in attribution["attribution"].items():
        print(f"  {name:<24} {stats['count']:>3}  ({stats['share'] * 100:>5.1f}% of failures)")

    failed_packages = [r["package"] for r in sweep["results"] if not r["valid"]]
    print(f"\ntesting M1: would a bigger context have rescued these {len(failed_packages)}?")
    intervention = context_intervention(sweep, graph, failed_packages)
    print(f"  fit in 512  : {intervention['fit_in_512']}/{intervention['tasks']}")
    print(f"  fit in 4096 : {intervention['fit_in_4096']}/{intervention['tasks']}")
    print(f"  rescued by a bigger context: "
          f"{intervention['would_be_rescued_by_bigger_context']}")

    print("\ncorrelates of failure")
    corr = correlations(sweep)
    for key, value in corr.items():
        if key != "note" and value is not None:
            print(f"  {key:<32} r = {value:+.4f}")

    report = {"attribution": attribution, "context_intervention": intervention,
              "correlations": corr,
              "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (OUTPUTS / "truncation.json").write_text(json.dumps(report, indent=2) + "\n")
    print("\n  wrote outputs/truncation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
