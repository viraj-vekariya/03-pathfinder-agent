"""HTTP surface: resolve install orders, search docs, and run the agent.

Three tiers of endpoint, and the separation is the point:

  /plan    - the ALGORITHM. Deterministic, exact, fast. What you would actually ship.
  /agent   - the LLM AGENT on the same task. Slower, and measurably worse past a
             certain depth.
  /compare - both, side by side, with the grade.

Most agent demos only expose the agent, which makes it impossible to see what the agent
costs you. Exposing the exact algorithm next to it is what turns this from a demo into
a measurement anyone can rerun.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

from agent.llm import get_llm
from agent.loop import PathfinderAgent
from agent.tools import Toolbox
from graph.algorithms import CycleError, a_star, detect_cycles, dijkstra
from graph.build import load
from graph.heuristics import (compare_expansions, make_depth_heuristic,
                              make_out_degree_heuristic, zero_heuristic)
from graph.planner import build_plan, grade_plan, packages_by_depth
from rag.corpus import build_corpus
from rag.embed import get_embedder
from rag.retrieve import Retriever
from rag.store import get_store

logging.basicConfig(level=os.environ.get("PATHFINDER_LOG_LEVEL", "INFO"))
log = logging.getLogger("pathfinder.api")

ROOT = Path(__file__).resolve().parent.parent
GRAPH_PATH = Path(os.environ.get("PATHFINDER_GRAPH", ROOT / "data" / "dependency_graph.json"))
UI = ROOT / "api" / "index.html"


class State:
    graph = None
    metadata: Dict = {}
    retriever: Optional[Retriever] = None
    agent: Optional[PathfinderAgent] = None
    llm = None
    ready = False
    error = ""


state = State()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state.graph, state.metadata = load(GRAPH_PATH)
        log.info("graph: %s", state.graph.stats())

        docs = build_corpus(state.graph, state.metadata)
        embedder = get_embedder("auto")
        vectors = embedder.encode([d.text for d in docs])
        store = get_store(embedder.dim)
        store.add([d.doc_id for d in docs], [d.package for d in docs],
                  [d.kind for d in docs], [d.text for d in docs],
                  [d.metadata for d in docs], vectors)
        state.retriever = Retriever(store, embedder, sorted(state.graph.nodes))
        log.info("indexed %d documents with %s", store.count(), embedder.name)

        state.llm = get_llm("auto")
        state.agent = PathfinderAgent(state.graph, state.llm,
                                      Toolbox(state.graph, state.retriever))
        state.ready = True
        log.info("agent ready: %s (%s), context %d",
                 state.llm.backend, state.llm.model, state.llm.context_limit)
    except Exception as exc:                     # noqa: BLE001
        state.error = str(exc)
        log.exception("startup failed; the algorithmic endpoints will still work")
    yield


app = FastAPI(title="Pathfinder Agent", version="1.0.0", lifespan=lifespan)


def _require_graph():
    if state.graph is None:
        raise HTTPException(503, f"graph not loaded: {state.error}")
    return state.graph


def _require_package(package: str) -> str:
    graph = _require_graph()
    if package not in graph.nodes:
        raise HTTPException(404, f"{package!r} is not in the graph "
                                 f"({len(graph.nodes)} packages known)")
    return package


@app.get("/health")
async def health():
    graph = state.graph
    return {
        "status": "ok" if state.ready else "degraded",
        "graph": graph.stats() if graph else None,
        "retrieval": state.retriever is not None,
        "llm": {"backend": state.llm.backend, "model": state.llm.model,
                "context_limit": state.llm.context_limit} if state.llm else None,
        "error": state.error,
    }


@app.get("/packages")
async def packages(q: str = "", limit: int = 40):
    graph = _require_graph()
    names = sorted(n for n in graph.nodes if q.lower() in n)
    return {"total": len(names), "packages": names[:limit]}


@app.get("/plan/{package}")
async def plan(package: str):
    """The exact algorithmic answer."""
    _require_package(package)
    started = time.perf_counter()
    result = build_plan(state.graph, package)
    return {**result.as_dict(), "ms": round((time.perf_counter() - started) * 1000, 3)}


@app.get("/path/{source}/{target}")
async def path(source: str, target: str, heuristic: str = "depth"):
    """Shortest dependency path, with A* vs Dijkstra expansion counts."""
    _require_package(source)
    _require_package(target)
    graph = state.graph

    heuristics = {
        "zero": zero_heuristic,
        "depth": make_depth_heuristic(graph, source),
        "out_degree": make_out_degree_heuristic(graph),
    }
    if heuristic not in heuristics:
        raise HTTPException(422, f"heuristic must be one of {sorted(heuristics)}")

    comparison = compare_expansions(graph, source, target, heuristics)
    _, _, dijkstra_expanded = dijkstra(graph, source, target)
    cost, route, expanded = a_star(graph, source, target, heuristics[heuristic])

    return {
        "source": source, "target": target,
        "reachable": cost != float("inf"),
        "cost": None if cost == float("inf") else cost,
        "path": route,
        "a_star_expanded": expanded,
        "dijkstra_expanded": dijkstra_expanded,
        "heuristic_comparison": comparison,
    }


@app.get("/cycles")
async def cycles(package: Optional[str] = None):
    graph = _require_graph()
    target = graph.subgraph_reachable_from([package]) if package else graph
    found = detect_cycles(target)
    return {"scope": package or "whole graph", "count": len(found), "cycles": found}


@app.get("/search")
async def search(q: str = Query(..., min_length=2), k: int = 6, rerank: bool = True):
    if state.retriever is None:
        raise HTTPException(503, "retrieval is not available")
    return state.retriever.retrieve(q, k=k, rerank=rerank).as_dict()


@app.get("/agent/{package}")
async def run_agent(package: str):
    """The agent's attempt, graded against the algorithm."""
    _require_package(package)
    if state.agent is None:
        raise HTTPException(503, "the agent is not available")
    result = state.agent.run(package)
    return {**result.as_dict(), "proposed": result.proposed}


@app.get("/compare/{package}")
async def compare(package: str):
    """Algorithm and agent, side by side. The endpoint the project is about."""
    _require_package(package)
    if state.agent is None:
        raise HTTPException(503, "the agent is not available")

    truth = build_plan(state.graph, package)
    run = state.agent.run(package)
    return {
        "package": package,
        "depth": truth.depth,
        "closure_size": truth.closure_size,
        "algorithm": {"order": truth.order, "exact": True},
        "agent": {
            "order": run.proposed,
            "valid": run.grade.valid,
            "coverage": round(run.grade.coverage, 4),
            "reason": run.grade.reason,
            "tools_used": run.tools_used,
            "input_truncated": run.any_truncation,
            "tokens_dropped": run.total_tokens_dropped,
            "max_prompt_tokens": run.max_prompt_tokens,
            "context_limit": run.context_limit,
            "seconds": round(run.seconds, 2),
        },
    }


@app.get("/depths")
async def depths():
    """How many packages exist at each depth. The sweep's sampling frame."""
    graph = _require_graph()
    buckets = packages_by_depth(graph, min_closure=2)
    return {str(d): {"count": len(v), "examples": [p for p, _ in v[-4:]]}
            for d, v in sorted(buckets.items())}


@app.get("/", response_class=HTMLResponse)
async def index():
    return UI.read_text() if UI.exists() else "<h1>Pathfinder</h1><p>UI not built.</p>"
