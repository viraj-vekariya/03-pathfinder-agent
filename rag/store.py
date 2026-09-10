"""Vector store with two backends behind one interface.

  * **pgvector** - Postgres with the pgvector extension. The deployment target, and
    what the API talks to in Docker.
  * **numpy** - an in-process exact-search index. Not a toy: at 822 documents an exact
    brute-force scan is FASTER than an approximate index, because building and probing
    an HNSW graph costs more than 822 dot products. It is also what makes the test
    suite and the evaluation runnable with no database.

Both return identical results, and a test asserts that. That matters more than it
sounds: an approximate index that silently returns different neighbours than the exact
one turns every downstream quality number into a measurement of the index rather than
of the retrieval.

WHY EXACT SEARCH IS THE RIGHT CHOICE HERE, stated so it is a decision rather than an
omission: brute force is O(N*d) = 822 x 384 = ~316k multiply-adds, well under a
millisecond in numpy. HNSW's advantage begins somewhere around 10^5-10^6 vectors. Using
an approximate index at this scale would add a dependency, a build step, a recall
parameter to tune, and a source of nondeterminism, in exchange for being slower.
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger("pathfinder.store")


@dataclass
class Hit:
    doc_id: str
    package: str
    kind: str
    text: str
    score: float
    metadata: Dict[str, object]

    def as_dict(self) -> Dict[str, object]:
        return {"doc_id": self.doc_id, "package": self.package, "kind": self.kind,
                "score": round(self.score, 4), "text": self.text}


class VectorStore(ABC):
    @abstractmethod
    def add(self, doc_ids, packages, kinds, texts, metadatas, vectors) -> None: ...

    @abstractmethod
    def search(self, query_vector: np.ndarray, k: int = 8,
               kind: Optional[str] = None) -> List[Hit]: ...

    @abstractmethod
    def count(self) -> int: ...


class NumpyStore(VectorStore):
    """Exact cosine search over unit-normalised vectors."""

    backend = "numpy"

    def __init__(self, dim: int):
        self.dim = dim
        self._vectors: Optional[np.ndarray] = None
        self._doc_ids: List[str] = []
        self._packages: List[str] = []
        self._kinds: List[str] = []
        self._texts: List[str] = []
        self._metadatas: List[Dict] = []

    def add(self, doc_ids, packages, kinds, texts, metadatas, vectors) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {vectors.shape[1]}")
        self._vectors = vectors if self._vectors is None else np.vstack([self._vectors, vectors])
        self._doc_ids += list(doc_ids)
        self._packages += list(packages)
        self._kinds += list(kinds)
        self._texts += list(texts)
        self._metadatas += list(metadatas)

    def search(self, query_vector: np.ndarray, k: int = 8,
               kind: Optional[str] = None) -> List[Hit]:
        if self._vectors is None or len(self._doc_ids) == 0:
            return []
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        # Both sides are unit vectors, so the dot product IS the cosine similarity.
        # Normalising at write time turns every query into one matrix multiply instead
        # of N divisions.
        scores = self._vectors @ query

        if kind is not None:
            mask = np.array([kd == kind for kd in self._kinds])
            scores = np.where(mask, scores, -np.inf)

        k = min(k, int(np.isfinite(scores).sum()))
        if k <= 0:
            return []
        # argpartition is O(N) against argsort's O(N log N); only the top k need sorting.
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]

        return [Hit(self._doc_ids[i], self._packages[i], self._kinds[i],
                    self._texts[i], float(scores[i]), self._metadatas[i]) for i in top]

    def count(self) -> int:
        return len(self._doc_ids)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, vectors=self._vectors,
            doc_ids=np.array(self._doc_ids), packages=np.array(self._packages),
            kinds=np.array(self._kinds), texts=np.array(self._texts, dtype=object),
            metadatas=np.array([json.dumps(m) for m in self._metadatas]))

    @classmethod
    def load(cls, path: Path) -> "NumpyStore":
        data = np.load(path, allow_pickle=True)
        store = cls(dim=data["vectors"].shape[1])
        store.add(data["doc_ids"].tolist(), data["packages"].tolist(),
                  data["kinds"].tolist(), data["texts"].tolist(),
                  [json.loads(m) for m in data["metadatas"]], data["vectors"])
        return store


class PgVectorStore(VectorStore):
    """Postgres + pgvector. The deployment backend.

    The index is IVFFlat rather than HNSW: it builds far faster on a small corpus and
    its recall is tunable at query time with `ivfflat.probes`, whereas HNSW's is baked
    in at build time. On a corpus this size either is overkill, which is the honest
    reason the numpy backend is the default.
    """

    backend = "pgvector"

    SCHEMA = """
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE TABLE IF NOT EXISTS documents (
        doc_id   TEXT PRIMARY KEY,
        package  TEXT NOT NULL,
        kind     TEXT NOT NULL,
        text     TEXT NOT NULL,
        metadata JSONB NOT NULL DEFAULT '{}',
        embedding vector(%(dim)s) NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_documents_kind ON documents(kind);
    """

    INDEX = """
    CREATE INDEX IF NOT EXISTS idx_documents_embedding
        ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = %(lists)s);
    """

    def __init__(self, dsn: str, dim: int):
        import psycopg                       # imported lazily: optional dependency
        self.psycopg = psycopg
        self.dsn = dsn
        self.dim = dim
        with psycopg.connect(dsn) as conn:
            conn.execute(self.SCHEMA % {"dim": dim})
            conn.commit()

    def add(self, doc_ids, packages, kinds, texts, metadatas, vectors) -> None:
        rows = [(d, p, k, t, json.dumps(m), "[" + ",".join(f"{x:.6f}" for x in v) + "]")
                for d, p, k, t, m, v in zip(doc_ids, packages, kinds, texts,
                                            metadatas, vectors)]
        with self.psycopg.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """INSERT INTO documents (doc_id, package, kind, text, metadata, embedding)
                       VALUES (%s,%s,%s,%s,%s,%s)
                       ON CONFLICT (doc_id) DO UPDATE
                         SET text = EXCLUDED.text, embedding = EXCLUDED.embedding""",
                    rows)
            # lists ~= sqrt(rows) is the pgvector guidance; too few makes each probe
            # scan a large partition, too many makes recall collapse.
            lists = max(1, int(len(rows) ** 0.5))
            conn.execute(self.INDEX % {"lists": lists})
            conn.commit()

    def search(self, query_vector, k: int = 8, kind: Optional[str] = None) -> List[Hit]:
        vec = "[" + ",".join(f"{x:.6f}" for x in np.asarray(query_vector).reshape(-1)) + "]"
        # `<=>` is pgvector's cosine DISTANCE, so similarity is 1 - distance.
        sql = ("SELECT doc_id, package, kind, text, metadata, 1 - (embedding <=> %s) "
               "AS score FROM documents")
        params: List[object] = [vec]
        if kind is not None:
            sql += " WHERE kind = %s"
            params.append(kind)
        sql += " ORDER BY embedding <=> %s LIMIT %s"
        params += [vec, k]

        with self.psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            return [Hit(r[0], r[1], r[2], r[3], float(r[5]),
                        r[4] if isinstance(r[4], dict) else json.loads(r[4] or "{}"))
                    for r in cur.fetchall()]

    def count(self) -> int:
        with self.psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM documents")
            return int(cur.fetchone()[0])


def get_store(dim: int, dsn: Optional[str] = None) -> VectorStore:
    """pgvector when a DSN is configured and reachable, numpy otherwise."""
    dsn = dsn or os.environ.get("PATHFINDER_PG_DSN")
    if dsn:
        try:
            return PgVectorStore(dsn, dim)
        except Exception as exc:                 # noqa: BLE001
            log.warning("pgvector unavailable (%s); using the numpy store", exc)
    return NumpyStore(dim)
