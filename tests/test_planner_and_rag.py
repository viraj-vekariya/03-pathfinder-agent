"""Plan grading, retrieval, and the heuristics' admissibility."""

import pytest

from graph.algorithms import Graph
from graph.heuristics import (compare_expansions, make_depth_heuristic,
                              make_out_degree_heuristic, verify_admissible,
                              zero_heuristic)
from graph.planner import build_plan, grade_plan, packages_by_depth
from rag.corpus import Document, build_corpus, stats
from rag.embed import HashedTfidf
from rag.retrieve import Retriever, infer_kind, mentioned_packages
from rag.store import NumpyStore


@pytest.fixture()
def graph():
    g = Graph()
    for a, b in [("app", "web"), ("app", "db"), ("web", "http"), ("db", "http"),
                 ("http", "sock"), ("web", "json")]:
        g.add_edge(a, b)
    return g


# -- grading -----------------------------------------------------------------

def test_a_correct_plan_is_valid(graph):
    plan = build_plan(graph, "app")
    assert grade_plan(graph, "app", plan.order).valid


def test_grading_accepts_ANY_valid_order_not_one_blessed_answer(graph):
    """A graph has many valid topological orders. Grading against a single reference
    would mark correct plans wrong."""
    a = ["sock", "http", "json", "web", "db", "app"]
    b = ["sock", "http", "db", "json", "web", "app"]
    assert grade_plan(graph, "app", a).valid
    assert grade_plan(graph, "app", b).valid


def test_a_reversed_plan_is_rejected_with_the_first_violation(graph):
    plan = build_plan(graph, "app")
    grade = grade_plan(graph, "app", list(reversed(plan.order)))
    assert not grade.valid
    assert grade.order_violations > 0
    assert grade.first_violation is not None


def test_a_missing_package_is_caught(graph):
    plan = build_plan(graph, "app")
    grade = grade_plan(graph, "app", [p for p in plan.order if p != "sock"])
    assert not grade.valid and "sock" in grade.missing


def test_a_duplicate_is_caught(graph):
    plan = build_plan(graph, "app")
    grade = grade_plan(graph, "app", plan.order + ["sock"])
    assert not grade.valid and grade.duplicates == ["sock"]


def test_invented_packages_are_reported_as_extra(graph):
    plan = build_plan(graph, "app")
    grade = grade_plan(graph, "app", plan.order + ["not-a-real-package"])
    assert "not-a-real-package" in grade.extra


def test_coverage_is_a_fraction_of_the_required_closure(graph):
    plan = build_plan(graph, "app")
    half = plan.order[: len(plan.order) // 2]
    grade = grade_plan(graph, "app", half)
    assert 0.0 < grade.coverage < 1.0


def test_packages_by_depth_excludes_dependency_free_packages(graph):
    """"install X" where X needs nothing is not a planning problem and would inflate
    accuracy at depth 0."""
    buckets = packages_by_depth(graph, min_closure=2)
    assert all(p != "sock" for rows in buckets.values() for p, _ in rows)


# -- heuristics --------------------------------------------------------------

def test_every_heuristic_is_admissible(graph):
    """REGRESSION: the depth heuristic originally returned |level difference|, which
    over-estimates whenever the goal is SHALLOWER than the node. verify_admissible
    caught it; this keeps it caught."""
    for name, h in {"zero": zero_heuristic,
                    "depth": make_depth_heuristic(graph, "app"),
                    "out_degree": make_out_degree_heuristic(graph)}.items():
        ok, violations = verify_admissible(graph, h, "sock")
        assert ok, f"{name} over-estimates: {violations[:2]}"


def test_an_inadmissible_heuristic_is_detected(graph):
    """The verifier must actually be able to fail, or it proves nothing."""
    ok, violations = verify_admissible(graph, lambda n, g: 99.0, "sock")
    assert not ok and violations


def test_all_admissible_heuristics_agree_on_the_optimal_cost(graph):
    comparison = compare_expansions(graph, "app", "sock", {
        "zero": zero_heuristic,
        "depth": make_depth_heuristic(graph, "app"),
        "out_degree": make_out_degree_heuristic(graph),
    })
    costs = {r["cost"] for r in comparison.values()}
    assert len(costs) == 1, f"heuristics disagree on cost: {costs}"


# -- retrieval ---------------------------------------------------------------

def test_kind_inference_is_not_fooled_by_question_words():
    """REGRESSION: 'what does flask depend on?' scored description=2 vs dependency=1,
    because the interrogatives outvoted the one topical word."""
    assert infer_kind("what does flask depend on?") == "dependency"
    assert infer_kind("what is pandas used for?") == "description"
    assert infer_kind("transitive dependencies of dvc") == "dependency"


def test_package_mentions_require_a_whole_name_not_a_substring():
    """'click' must not match 'clickhouse-driver' - substring matching quietly
    poisons the results."""
    known = ["click", "clickhouse-driver", "scikit-learn"]
    assert mentioned_packages("what does click need", known) == ["click"]
    assert "scikit-learn" in mentioned_packages("tell me about scikit learn", known)


def test_corpus_splits_description_from_dependency(graph):
    docs = build_corpus(graph, {})
    kinds = stats(docs)["by_kind"]
    assert kinds["description"] == kinds["dependency"] == len(graph.nodes)


def test_retrieval_finds_the_right_package(graph):
    docs = build_corpus(graph, {})
    embedder = HashedTfidf().fit([d.text for d in docs])
    vectors = embedder.encode([d.text for d in docs])
    store = NumpyStore(embedder.dim)
    store.add([d.doc_id for d in docs], [d.package for d in docs],
              [d.kind for d in docs], [d.text for d in docs],
              [d.metadata for d in docs], vectors)
    retriever = Retriever(store, embedder, sorted(graph.nodes))
    hits = retriever.retrieve("what does web depend on", k=3).hits
    assert any(h.package == "web" for h in hits)


def test_reranking_promotes_the_named_package(graph):
    docs = build_corpus(graph, {})
    embedder = HashedTfidf().fit([d.text for d in docs])
    store = NumpyStore(embedder.dim)
    store.add([d.doc_id for d in docs], [d.package for d in docs],
              [d.kind for d in docs], [d.text for d in docs],
              [d.metadata for d in docs], embedder.encode([d.text for d in docs]))
    retriever = Retriever(store, embedder, sorted(graph.nodes))
    reranked = retriever.retrieve("what does db depend on", k=2, rerank=True)
    assert reranked.hits[0].package == "db"


def test_store_filters_by_kind(graph):
    docs = build_corpus(graph, {})
    embedder = HashedTfidf().fit([d.text for d in docs])
    store = NumpyStore(embedder.dim)
    store.add([d.doc_id for d in docs], [d.package for d in docs],
              [d.kind for d in docs], [d.text for d in docs],
              [d.metadata for d in docs], embedder.encode([d.text for d in docs]))
    hits = store.search(embedder.encode(["dependencies"])[0], k=5, kind="dependency")
    assert hits and all(h.kind == "dependency" for h in hits)
