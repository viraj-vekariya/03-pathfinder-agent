"""Build a real dependency graph from the live PyPI JSON API.

Real data, not a fixture. The PyPI JSON API is public, needs no credentials, and
returns each release's declared `requires_dist`. That is what makes the depth numbers
in the evaluation real dependency depths rather than depths of a graph invented to
produce them.

Three things this has to get right, and each is a place a naive version breaks:

1. **Requirement strings are not package names.** `charset_normalizer<4,>=2` and
   `urllib3 (>=1.26,<3)` and `pytest; extra == "test"` all have to become - or not
   become - a node. The extras marker is the important one: pulling in every optional
   test dependency roughly triples the graph and none of it is installed by default.

2. **Fetches must be cached.** The evaluation runs many times over the same packages,
   and hammering PyPI for data that has not changed is both slow and rude.

3. **The traversal must be bounded.** Dependency closures can be large and the API can
   be slow; an unbounded BFS on a bad day never finishes.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .algorithms import Graph

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = Path(os.environ.get("PATHFINDER_CACHE", ROOT / "data" / "pypi_cache"))
USER_AGENT = "pathfinder-agent/1.0 (placement project; +https://pypi.org)"

# A requirement string starts with the distribution name, then optional extras in
# brackets, then version specifiers, then an environment marker after ';'.
#   "charset_normalizer<4,>=2"        -> charset_normalizer
#   "urllib3 (>=1.26,<3)"             -> urllib3
#   "pytest; extra == 'test'"         -> skipped, it is an extra
_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def normalise(name: str) -> str:
    """PEP 503 normalisation. `Flask`, `flask` and `FLASK` are one package, and
    `zope.interface` and `zope-interface` are too. Without this the graph gets
    duplicate nodes that never join up."""
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_requirement(spec: str) -> Optional[str]:
    """Extract the dependency name, or None if it should not be an edge."""
    if ";" in spec:
        _, marker = spec.split(";", 1)
        # Optional extras are not installed by default. Including them inflates the
        # graph with test and docs tooling that no user of the package ever gets.
        if "extra" in marker:
            return None
    match = _NAME_RE.match(spec)
    return normalise(match.group(1)) if match else None


def _cache_path(package: str) -> Path:
    return CACHE_DIR / f"{normalise(package)}.json"


def fetch_metadata(package: str, timeout: float = 15.0,
                   offline: bool = False) -> Optional[Dict]:
    """Package metadata, cache-first.

    A miss in offline mode returns None rather than raising: the evaluation must be
    runnable on a machine with no network using whatever was cached, and a hard failure
    would make one missing leaf package abort a whole sweep.
    """
    path = _cache_path(package)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            path.unlink(missing_ok=True)      # a truncated cache file is not a fact

    if offline:
        return None

    url = f"https://pypi.org/pypi/{package}/json"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            # Genuinely absent - cache the absence so the sweep does not re-ask.
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"__missing__": True}))
            return None
        raise
    except (urllib.error.URLError, TimeoutError):
        return None

    info = payload.get("info", {})
    slim = {
        "name": normalise(info.get("name", package)),
        "version": info.get("version", ""),
        "summary": (info.get("summary") or "")[:400],
        "requires_dist": info.get("requires_dist") or [],
        "requires_python": info.get("requires_python") or "",
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(slim, indent=2))
    return slim


def build_graph(roots: Iterable[str], max_nodes: int = 400, max_depth: int = 6,
                offline: bool = False, verbose: bool = False
                ) -> Tuple[Graph, Dict[str, Dict]]:
    """BFS the real dependency closure of `roots`.

    Breadth-first rather than depth-first so that `max_nodes` truncates the graph at its
    OUTER edge: a depth-first walk that hits the cap leaves one branch fully explored and
    the rest untouched, which is a badly skewed graph. BFS leaves a uniform frontier.

    Returns the graph and the per-package metadata, which the RAG corpus is built from.
    """
    graph = Graph()
    metadata: Dict[str, Dict] = {}
    seen: Set[str] = set()
    queue: deque = deque((normalise(r), 0) for r in roots)

    while queue and len(seen) < max_nodes:
        package, depth = queue.popleft()
        if package in seen or depth > max_depth:
            continue
        seen.add(package)
        graph.add_node(package)

        info = fetch_metadata(package, offline=offline)
        if not info or info.get("__missing__"):
            continue
        metadata[package] = info

        if verbose:
            print(f"  {'  ' * depth}{package} {info.get('version','')}", file=sys.stderr)

        for spec in info["requires_dist"]:
            dependency = parse_requirement(spec)
            if not dependency or dependency == package:
                continue
            graph.add_edge(package, dependency)
            if dependency not in seen:
                queue.append((dependency, depth + 1))

    return graph, metadata


def save(graph: Graph, metadata: Dict[str, Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "nodes": sorted(graph.nodes),
        "edges": {k: sorted(v) for k, v in sorted(graph.edges.items()) if v},
        "metadata": metadata,
        "stats": graph.stats(),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2) + "\n")


def load(path: Path) -> Tuple[Graph, Dict[str, Dict]]:
    payload = json.loads(path.read_text())
    graph = Graph()
    for node in payload["nodes"]:
        graph.add_node(node)
    for src, dsts in payload["edges"].items():
        for dst in dsts:
            graph.add_edge(src, dst)
    return graph, payload.get("metadata", {})


DEFAULT_ROOTS = [
    # Chosen for DEPTH RANGE, not popularity. The evaluation's independent variable is
    # dependency depth, so the corpus needs shallow leaves and deep trunks alike.
    # An earlier root set of ordinary libraries topped out at depth 4, which is a real
    # property of modern pip packages once extras are excluded - and useless as a sweep
    # axis. These add orchestration and ML stacks, which are genuinely deep.
    "requests", "flask", "fastapi", "pandas", "scikit-learn", "sqlalchemy",
    "celery", "black", "pytest", "rich", "httpx", "pydantic", "jinja2", "click",
    "apache-airflow", "dask", "great-expectations", "transformers", "datasets",
    "jupyterlab", "mlflow", "streamlit", "dvc", "prefect", "ray", "langchain",
    "boto3", "google-cloud-storage", "opentelemetry-sdk", "sphinx", "poetry",
]

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=DEFAULT_ROOTS)
    ap.add_argument("--max-nodes", type=int, default=400)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "data" / "dependency_graph.json"))
    args = ap.parse_args()

    started = time.time()
    graph, metadata = build_graph(args.roots, max_nodes=args.max_nodes,
                                  offline=args.offline)
    save(graph, metadata, Path(args.out))
    print(json.dumps({"roots": len(args.roots), **graph.stats(),
                      "with_metadata": len(metadata),
                      "seconds": round(time.time() - started, 1),
                      "out": args.out}, indent=2))
