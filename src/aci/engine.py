"""CodeSearchEngine: loads version indexes on demand and answers queries, optionally across versions."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import load_config
from .errors import EmptyQueryError, IndexNotFoundError
from .indexing.builder import read_refs, state_dir
from .indexing.store import VersionIndex
from .models import SearchResult
from .pipeline import PipelineOptions, SearchPipeline
from .ranking.dedup import dedup_versions
from .retrieval.dense import load_embedder


class CodeSearchEngine:
    def __init__(self, repo: str | Path, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        name = Path(repo).expanduser().resolve().name if (Path(repo).exists() or "/" in str(repo)) else str(repo)
        self.name = name
        self.root = state_dir(self.cfg, name)
        if not (self.root / "refs.json").exists():
            raise IndexNotFoundError(f"No index found for repository '{name}' under {self.root}. Run: python scripts/build_index.py --repo <path>")
        self._pipes: dict[str, SearchPipeline] = {}
        self._embedder = None
        self.last_timings: dict[str, float] = {}

    def versions(self) -> list[dict]:
        v = read_refs(self.root)["versions"]
        return [{"label": k, **val} for k, val in sorted(v.items(), key=lambda kv: kv[1]["timestamp"])]

    def resolve(self, ref: str | None) -> str:
        vs = self.versions()
        if not vs:
            raise IndexNotFoundError(f"Repository '{self.name}' has no indexed versions.")
        if ref in (None, "", "latest", "HEAD"):
            return vs[-1]["label"]
        for v in vs:
            if ref == v["label"] or v["commit"] == ref or v["commit"].startswith(ref):
                return v["label"]
        raise IndexNotFoundError(f"Version '{ref}' is not indexed. Available: {[v['label'] for v in vs]}")

    def _pipe(self, label: str) -> SearchPipeline:
        if label not in self._pipes:
            if self._embedder is None:
                self._embedder = load_embedder(self.cfg, self.root)
            vi = VersionIndex.load(self.root / label, self.cfg, self._embedder)
            self._pipes[label] = SearchPipeline(vi, self.cfg)
        return self._pipes[label]

    def search(self, query: str, version: str | None = None, top_k: int = 10, dedup: bool | None = None,
               history: bool = False, options: PipelineOptions | None = None) -> list[SearchResult]:
        """version: a label/commit, None for the latest, or "all" to search every indexed version.
        history=True keeps every version of a unit instead of collapsing them."""
        if not query or not query.strip():
            raise EmptyQueryError("Query is empty. Provide a natural-language question or identifier.")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if version == "all":
            merged: list[SearchResult] = []
            agg: dict[str, float] = {}
            for v in self.versions():
                res, tm = self._pipe(v["label"]).search(query, top_k * 3, options)
                for r in res:
                    r.vec = self._pipe(v["label"]).vi.vectors[self._index_of(v["label"], r)]
                merged += res
                for k, x in tm.items():
                    agg[k] = agg.get(k, 0.0) + x
            self.last_timings = agg
            merged.sort(key=lambda r: -r.score)
            if not history and (self.cfg["dedup"]["enabled"] if dedup is None else dedup):
                merged = dedup_versions(merged, [r.vec for r in merged], self.cfg["dedup"]["similarity_threshold"])
            for i, r in enumerate(merged):
                r.rank = i + 1
            return merged[:top_k]
        label = self.resolve(version)
        res, self.last_timings = self._pipe(label).search(query, top_k, options)
        return res

    def _index_of(self, label: str, r: SearchResult) -> int:
        pipe = self._pipe(label)
        m = getattr(pipe, "_pos", None)
        if m is None:
            m = pipe._pos = {(c.file, c.qualname, c.start_line): i for i, c in enumerate(pipe.vi.chunks)}
        return m[(r.chunk.file, r.chunk.qualname, r.chunk.start_line)]
