"""Embeddings, computed locally.

Two backends, and the fallback is not decorative:

  * **MiniLM** (sentence-transformers/all-MiniLM-L6-v2, 22.7M params, 384 dims) - a
    real sentence embedding model, run through plain `transformers` with mean pooling.
  * **Hashed TF-IDF** - deterministic, dependency-light, no download. It is what runs
    when the model cannot be fetched, and it is also the CONTROL: retrieval quality is
    reported for both, so the evaluation can say whether the neural embedding actually
    earned its 90MB rather than assuming it did.

Mean pooling over the last hidden state, masked by attention. Using the [CLS] token
instead is a common mistake with this family: MiniLM's sentence-level behaviour comes
from a mean-pooling training objective, and [CLS] is not trained to carry it.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

log = logging.getLogger("pathfinder.embed")

MODEL_NAME = os.environ.get("PATHFINDER_EMBED_MODEL",
                            "sentence-transformers/all-MiniLM-L6-v2")
HASH_DIM = 512
_TOKEN_RE = re.compile(r"[a-z0-9]+")


class HashedTfidf:
    """Deterministic hashing vectoriser with IDF weighting.

    Hashing rather than a fitted vocabulary so a new package at query time does not
    need a refit. Collisions are accepted: at 512 dimensions over a few thousand terms
    they are rare enough not to matter, and the alternative is state that has to be
    persisted alongside the index.
    """

    name = "hashed-tfidf"
    dim = HASH_DIM

    def __init__(self, dim: int = HASH_DIM):
        self.dim = dim
        self._idf: Optional[np.ndarray] = None

    @staticmethod
    def _tokens(text: str) -> List[str]:
        # Split on underscores and hyphens too: "scikit-learn" must share a token with
        # a query saying "scikit learn", or package names never match prose.
        return _TOKEN_RE.findall(text.lower().replace("-", " ").replace("_", " "))

    def _bucket(self, token: str) -> int:
        return int(hashlib.md5(token.encode()).hexdigest()[:8], 16) % self.dim

    def fit(self, texts: Sequence[str]) -> "HashedTfidf":
        df = np.zeros(self.dim)
        for text in texts:
            for bucket in {self._bucket(t) for t in self._tokens(text)}:
                df[bucket] += 1
        n = max(1, len(texts))
        # Smoothed IDF. Without the +1 terms, a bucket present in every document gives
        # log(1)=0 and a bucket in none divides by zero.
        self._idf = np.log((1 + n) / (1 + df)) + 1.0
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        idf = self._idf if self._idf is not None else np.ones(self.dim)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in self._tokens(text):
                out[i, self._bucket(token)] += 1.0
        out *= idf
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)      # unit vectors: cosine == dot product


class MiniLMEmbedder:
    name = MODEL_NAME
    dim = 384

    def __init__(self, model_name: str = MODEL_NAME, batch_size: int = 64):
        from transformers import AutoModel, AutoTokenizer
        import torch

        self.torch = torch
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.dim = self.model.config.hidden_size

    def fit(self, texts: Sequence[str]) -> "MiniLMEmbedder":
        return self          # nothing to fit; the model is pretrained

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        torch = self.torch
        vectors: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = list(texts[start:start + self.batch_size])
                encoded = self.tokenizer(batch, padding=True, truncation=True,
                                         max_length=256, return_tensors="pt")
                output = self.model(**encoded).last_hidden_state

                # Masked mean pooling. Padding tokens must not contribute, or short
                # texts in a batch with long ones get their embeddings diluted by
                # however much padding they happened to receive.
                mask = encoded["attention_mask"].unsqueeze(-1).float()
                pooled = (output * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                vectors.append(pooled.cpu().numpy().astype(np.float32))
        return np.vstack(vectors) if vectors else np.zeros((0, self.dim), dtype=np.float32)


def get_embedder(prefer: str = "auto"):
    """Return an embedder. `prefer` is "auto" | "minilm" | "hashed"."""
    if prefer == "hashed":
        return HashedTfidf()
    try:
        return MiniLMEmbedder()
    except Exception as exc:                     # noqa: BLE001
        if prefer == "minilm":
            raise
        log.warning("MiniLM unavailable (%s); falling back to hashed TF-IDF", exc)
        return HashedTfidf()
