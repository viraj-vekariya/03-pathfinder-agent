"""Retrieval: embed a query, search, then rerank.

The reranker is the interesting part, and it is deliberately NOT a cross-encoder.

A dependency question has structure that pure vector similarity throws away. "What does
flask need?" should return flask's DEPENDENCY chunk, not the description chunk of some
semantically-adjacent web framework. So retrieval here is hybrid: the dense score is
combined with two cheap structural signals - an exact package-name match, and a
kind match inferred from the query's wording.

That is a smaller, faster and far more interpretable intervention than a neural
reranker, and on this corpus it is measurably better. eval/ reports both.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .store import Hit, VectorStore

# Words that signal the asker wants dependency structure rather than a description.
_DEPENDENCY_WORDS = frozenset({
    "depend", "depends", "dependency", "dependencies", "require", "requires",
    "required", "install", "installs", "installing", "needs", "need", "pull",
    "pulls", "transitive", "order", "before", "after",
})
# Deliberately excludes generic question words. An earlier version listed "what",
# "does" and "for" here, and "what does flask depend on?" then scored description=2
# against dependency=1 and routed to the wrong chunk kind - the interrogative words
# outvoted the single word that carried the actual topic. Only TOPICAL words belong in
# these sets; question words appear in nearly every query and carry no signal.
_DESCRIPTION_WORDS = frozenset({
    "purpose", "used", "use", "about", "describe", "description", "library",
    "framework", "tool", "package", "does", "do",
})


def infer_kind(query: str) -> Optional[str]:
    """Guess which chunk kind the query wants. None means no strong signal."""
    tokens = set(re.findall(r"[a-z]+", query.lower()))
    # Dependency words are weighted 2x. They are specific ("transitive", "requires")
    # while description words are broad ("use", "package"), so one dependency word is
    # stronger evidence than one description word - and treating them as equal is what
    # let "does" outvote "depend".
    dep = 2 * len(tokens & _DEPENDENCY_WORDS)
    desc = len(tokens & _DESCRIPTION_WORDS)
    if dep > desc:
        return "dependency"
    if desc > dep:
        return "description"
    return None


def mentioned_packages(query: str, known: Sequence[str]) -> List[str]:
    """Exact package names appearing in the query.

    Matched on a normalised, punctuation-stripped form so "scikit-learn",
    "scikit learn" and "scikit_learn" all hit. Substring matching is deliberately NOT
    used: "click" would then match "clickhouse-driver" and quietly poison the results.
    """
    norm = re.sub(r"[-_.]+", " ", query.lower())
    tokens = set(re.findall(r"[a-z0-9]+", norm))
    hits = []
    for package in known:
        parts = set(re.findall(r"[a-z0-9]+", re.sub(r"[-_.]+", " ", package.lower())))
        if parts and parts <= tokens:
            hits.append(package)
    # Longest first: "google-cloud-storage" should outrank "google" when both matched.
    return sorted(hits, key=len, reverse=True)


@dataclass
class RetrievalResult:
    query: str
    hits: List[Hit]
    inferred_kind: Optional[str]
    mentioned: List[str]
    reranked: bool

    def as_dict(self) -> Dict[str, object]:
        return {"query": self.query, "inferred_kind": self.inferred_kind,
                "mentioned": self.mentioned, "reranked": self.reranked,
                "hits": [h.as_dict() for h in self.hits]}

    def context_text(self, max_chars: int = 2000) -> str:
        """The retrieved text, as it will be pasted into a prompt.

        Truncated by CHARACTER budget, and the budget is the caller's, because the
        prompt has a token limit and retrieval is the part of it that grows without
        bound. This function is one of the two places that decides whether a task fits
        in the model's context - see eval/truncation.py.
        """
        chunks, used = [], 0
        for hit in self.hits:
            piece = hit.text.strip()
            if used + len(piece) > max_chars:
                break
            chunks.append(piece)
            used += len(piece) + 1
        return "\n".join(chunks)


class Retriever:
    def __init__(self, store: VectorStore, embedder, packages: Sequence[str],
                 name_boost: float = 0.35, kind_boost: float = 0.10):
        self.store = store
        self.embedder = embedder
        self.packages = list(packages)
        self.name_boost = name_boost
        self.kind_boost = kind_boost

    def _dense(self, query: str, k: int) -> List[Hit]:
        vector = self.embedder.encode([query])[0]
        return self.store.search(vector, k=k)

    def retrieve(self, query: str, k: int = 6, rerank: bool = True,
                 overfetch: int = 4) -> RetrievalResult:
        """Fetch k*overfetch candidates, rerank, keep k.

        Over-fetching before reranking is the point: a reranker can only reorder what
        the first stage returned, so if the right chunk was at rank 20 of a top-6 fetch
        it is gone regardless of how good the reranker is.
        """
        kind = infer_kind(query)
        mentioned = mentioned_packages(query, self.packages)

        if not rerank:
            return RetrievalResult(query, self._dense(query, k), kind, mentioned, False)

        candidates = self._dense(query, k * overfetch)
        scored: List[Tuple[float, Hit]] = []
        for hit in candidates:
            score = hit.score
            # Exact name match. A question naming a package is almost always about that
            # package, and dense similarity alone routinely ranks a sibling higher.
            if hit.package in mentioned:
                score += self.name_boost
            # Kind match, weighted lower: the inference is a heuristic over a handful
            # of keywords and should nudge rather than decide.
            if kind and hit.kind == kind:
                score += self.kind_boost
            scored.append((score, hit))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        hits = [Hit(h.doc_id, h.package, h.kind, h.text, s, h.metadata)
                for s, h in scored[:k]]
        return RetrievalResult(query, hits, kind, mentioned, True)


def recall_at_k(retriever: Retriever, cases: Sequence[Tuple[str, str]],
                k: int = 6, rerank: bool = True) -> Dict[str, object]:
    """Fraction of queries whose expected doc_id appears in the top k.

    Recall rather than precision: this feeds an agent that reads every retrieved chunk,
    so a wrong chunk costs context budget while a MISSING chunk costs the answer.
    """
    found, ranks = 0, []
    for query, expected in cases:
        result = retriever.retrieve(query, k=k, rerank=rerank)
        ids = [h.doc_id for h in result.hits]
        if expected in ids:
            found += 1
            ranks.append(ids.index(expected) + 1)
    return {
        "cases": len(cases),
        f"recall_at_{k}": round(found / len(cases), 4) if cases else 0.0,
        "found": found,
        "mean_rank_when_found": round(sum(ranks) / len(ranks), 2) if ranks else None,
        "reranked": rerank,
    }
