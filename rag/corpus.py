"""Build a retrieval corpus from real package documentation.

Source is the metadata already fetched from PyPI - each package's summary, declared
Python requirement, version and dependency list. Real text written by real maintainers,
not generated filler, which matters because retrieval quality on synthetic text tells
you nothing about retrieval quality.

The chunking decision is the substantive one. Two kinds of document are produced per
package, because the agent asks two different kinds of question:

  * a DESCRIPTION chunk - what is this package for? Answers "which package does X?"
  * a DEPENDENCY chunk  - what does it need, and what needs it? Answers "what does
                          installing X pull in?"

Putting both in one chunk would make every embedding a blend of two topics and both
queries would retrieve it equally badly. This is the most common RAG failure and it is
a chunking decision, not a model decision.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from graph.algorithms import Graph


@dataclass
class Document:
    doc_id: str
    package: str
    kind: str            # "description" | "dependency"
    text: str
    metadata: Dict[str, object]

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def _dependents_of(graph: Graph, package: str) -> List[str]:
    return sorted(src for src, dsts in graph.edges.items() if package in dsts)


def build_corpus(graph: Graph, metadata: Dict[str, Dict]) -> List[Document]:
    docs: List[Document] = []

    for package in sorted(graph.nodes):
        info = metadata.get(package, {})
        summary = (info.get("summary") or "").strip()
        version = info.get("version", "")
        requires_python = info.get("requires_python", "")
        deps = sorted(graph.neighbours(package))
        dependents = _dependents_of(graph, package)

        # -- description chunk ------------------------------------------------
        # Written as a sentence rather than a field dump. Embedding models are trained
        # on prose; "flask: A simple framework" embeds closer to a natural-language
        # query than "name=flask summary=A simple framework" does.
        if summary:
            text = f"{package} is a Python package. {summary}"
        else:
            text = f"{package} is a Python package."
        if version:
            text += f" The current version is {version}."
        if requires_python:
            text += f" It requires Python {requires_python}."
        docs.append(Document(
            doc_id=f"{package}::description",
            package=package,
            kind="description",
            text=text,
            metadata={"version": version, "n_dependencies": len(deps),
                      "n_dependents": len(dependents)},
        ))

        # -- dependency chunk -------------------------------------------------
        if deps:
            dep_text = (f"Installing {package} also installs its dependencies: "
                        f"{', '.join(deps)}. {package} directly requires "
                        f"{len(deps)} package{'s' if len(deps) != 1 else ''}.")
        else:
            dep_text = (f"{package} has no dependencies. Installing {package} installs "
                        f"nothing else.")
        if dependents:
            shown = dependents[:8]
            dep_text += (f" It is required by {', '.join(shown)}"
                         f"{f' and {len(dependents) - 8} others' if len(dependents) > 8 else ''}.")
        docs.append(Document(
            doc_id=f"{package}::dependency",
            package=package,
            kind="dependency",
            text=dep_text,
            metadata={"dependencies": deps, "dependents": dependents[:20]},
        ))

    return docs


def save(docs: Iterable[Document], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(d.as_dict()) for d in docs) + "\n")


def load(path: Path) -> List[Document]:
    """JSONL, so a large corpus streams instead of being parsed as one object."""
    docs = []
    with path.open() as fh:
        for line in fh:
            if line.strip():
                docs.append(Document(**json.loads(line)))
    return docs


def stats(docs: List[Document]) -> Dict[str, object]:
    lengths = [len(d.text.split()) for d in docs]
    kinds: Dict[str, int] = {}
    for d in docs:
        kinds[d.kind] = kinds.get(d.kind, 0) + 1
    return {
        "documents": len(docs),
        "packages": len({d.package for d in docs}),
        "by_kind": kinds,
        "mean_words": round(sum(lengths) / len(lengths), 1) if lengths else 0,
        "max_words": max(lengths) if lengths else 0,
    }
