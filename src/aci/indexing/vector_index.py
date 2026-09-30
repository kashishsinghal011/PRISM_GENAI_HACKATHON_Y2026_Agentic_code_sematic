"""ANN vector index (inner product on L2-normalised vectors == cosine).

FAISS HNSW for large indexes, exact FAISS flat below `flat_threshold` (exact is both cheaper and
better on tiny corpora), and a pure-NumPy fallback if faiss is not installed.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)
try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - exercised only when faiss is missing
    faiss = None


class VectorIndex:
    def __init__(self, dim: int, ann: str = "hnsw", flat_threshold: int = 2000, hnsw_m: int = 32,
                 ef_construction: int = 100, ef_search: int = 128):
        self.dim, self.ann, self.flat_threshold = dim, ann, flat_threshold
        self.m, self.efc, self.efs = hnsw_m, ef_construction, ef_search
        self.backend = "none"
        self._index = None
        self._vectors: np.ndarray | None = None

    def build(self, vectors: np.ndarray) -> "VectorIndex":
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"Expected vectors of shape (n, {self.dim}), got {vectors.shape}")
        self._vectors = vectors
        if faiss is None:
            self.backend = "numpy-exact"
        elif self.ann == "hnsw" and len(vectors) > self.flat_threshold:
            idx = faiss.IndexHNSWFlat(self.dim, self.m, faiss.METRIC_INNER_PRODUCT)
            idx.hnsw.efConstruction = self.efc
            idx.add(vectors)
            idx.hnsw.efSearch = self.efs
            self._index, self.backend = idx, "faiss-hnsw"
        else:
            idx = faiss.IndexFlatIP(self.dim)
            idx.add(vectors)
            self._index, self.backend = idx, "faiss-flat"
        return self

    def __len__(self) -> int:
        return 0 if self._vectors is None else len(self._vectors)

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        q = np.ascontiguousarray(np.atleast_2d(queries), dtype=np.float32)
        k = max(1, min(k, len(self)))
        if self._index is not None:
            if self.backend == "faiss-hnsw":
                self._index.hnsw.efSearch = max(self.efs, k)
            scores, ids = self._index.search(q, k)
            return scores, ids
        sims = q @ self._vectors.T
        ids = np.argsort(-sims, axis=1)[:, :k]
        return np.take_along_axis(sims, ids, axis=1), ids

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        if self._index is not None:
            faiss.write_index(self._index, str(d / "faiss.index"))
        (d / "vindex.txt").write_text(self.backend)

    @classmethod
    def load(cls, directory: str | Path, vectors: np.ndarray, **kw) -> "VectorIndex":
        d = Path(directory)
        vi = cls(vectors.shape[1], **kw)
        vi._vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        f = d / "faiss.index"
        if faiss is not None and f.exists():
            vi._index = faiss.read_index(str(f))
            vi.backend = (d / "vindex.txt").read_text().strip() if (d / "vindex.txt").exists() else "faiss"
            if "hnsw" in vi.backend:
                vi._index.hnsw.efSearch = vi.efs
        else:
            vi.build(vectors)
        return vi
