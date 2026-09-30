"""BM25 (sparse-matrix implementation, fast on 100K+ chunks) and an exact-identifier index."""
from __future__ import annotations

import math
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from ..models import Chunk


class BM25Index:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.vocab: dict[str, int] = {}
        self.W: sp.csc_matrix | None = None
        self.n_docs = 0

    def build(self, docs: list[list[str]]) -> "BM25Index":
        self.n_docs = len(docs)
        vocab: dict[str, int] = {}
        rows, cols, vals = [], [], []
        dl = np.zeros(len(docs), dtype=np.float32)
        for i, toks in enumerate(docs):
            dl[i] = len(toks)
            for t, c in Counter(toks).items():
                j = vocab.setdefault(t, len(vocab))
                rows.append(i); cols.append(j); vals.append(c)
        tf = sp.csr_matrix((np.asarray(vals, dtype=np.float32), (rows, cols)), shape=(len(docs), max(1, len(vocab))))
        df = np.bincount(tf.indices, minlength=tf.shape[1]).astype(np.float32)
        idf = np.log(1.0 + (len(docs) - df + 0.5) / (df + 0.5)).astype(np.float32)
        avgdl = float(dl.mean()) if len(docs) else 1.0
        tf = tf.tocoo()
        denom = tf.data + self.k1 * (1 - self.b + self.b * dl[tf.row] / max(avgdl, 1e-9))
        data = idf[tf.col] * tf.data * (self.k1 + 1) / denom
        self.W = sp.csc_matrix((data.astype(np.float32), (tf.row, tf.col)), shape=tf.shape)
        self.vocab = vocab
        return self

    def score_all(self, terms: dict[str, float]) -> np.ndarray:
        """BM25 score for every document given weighted query terms (weight = query-term frequency)."""
        out = np.zeros(self.n_docs, dtype=np.float32)
        cols, w = [], []
        for t, wt in terms.items():
            j = self.vocab.get(t)
            if j is not None:
                cols.append(j); w.append(wt)
        if cols:
            out = np.asarray(self.W[:, cols] @ np.asarray(w, dtype=np.float32)).ravel()
        return out

    def search(self, terms: dict[str, float], k: int) -> tuple[np.ndarray, np.ndarray]:
        s = self.score_all(terms)
        k = min(k, len(s))
        if k <= 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.int64)
        ids = np.argpartition(-s, k - 1)[:k]
        ids = ids[np.argsort(-s[ids])]
        keep = s[ids] > 0
        return s[ids][keep], ids[keep]

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"k1": self.k1, "b": self.b, "vocab": self.vocab, "W": self.W, "n": self.n_docs}, f, protocol=4)

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        with open(path, "rb") as f:
            d = pickle.load(f)
        o = cls(d["k1"], d["b"])
        o.vocab, o.W, o.n_docs = d["vocab"], d["W"], d["n"]
        return o


class IdentifierIndex:
    """name -> chunk ids. `defined` = the chunk defines that function/class; `called` = it calls it."""

    def __init__(self) -> None:
        self.defined: dict[str, list[int]] = {}
        self.called: dict[str, list[int]] = {}
        self.file_stem: dict[str, list[int]] = {}

    def build(self, chunks: list[Chunk]) -> "IdentifierIndex":
        for i, c in enumerate(chunks):
            for n in {c.function.lower(), c.class_name.lower(), c.parent.lower()} - {""}:
                self.defined.setdefault(n, []).append(i)
            for n in {x.lower() for x in c.calls}:
                self.called.setdefault(n, []).append(i)
            stem = c.file.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()
            self.file_stem.setdefault(stem, []).append(i)
        return self

    def search(self, names: list[str], files: list[str] | None = None) -> dict[int, float]:
        """Exact match scores: defines the name = 1.0, file has that stem = 0.8, calls it = 0.5."""
        scores: dict[int, float] = {}
        for n in names:
            for i in self.defined.get(n, ()):
                scores[i] = max(scores.get(i, 0.0), 1.0)
            for i in self.called.get(n, ()):
                scores[i] = max(scores.get(i, 0.0), 0.5)
        for f in files or []:
            stem = f.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()
            for i in self.file_stem.get(stem, ()):
                scores[i] = max(scores.get(i, 0.0), 0.8)
        return scores
