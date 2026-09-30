"""SearchPipeline: query -> preprocess -> classify -> hybrid retrieval -> rerank -> top-k (one version)."""
from __future__ import annotations

import time
from dataclasses import dataclass

from .indexing.store import VersionIndex
from .models import SearchResult
from .query.classifier import classify_query
from .query.preprocess import preprocess_query
from .ranking.ranker import Ranker
from .retrieval.bm25 import BM25Retriever
from .retrieval.dense import DenseRetriever
from .retrieval.hybrid import HybridRetriever


@dataclass
class PipelineOptions:
    """Switches used by the ablation study. Defaults = full system."""
    use_dense: bool = True
    use_bm25: bool = True
    use_identifier: bool = True
    use_rerank: bool = True
    use_expansion: bool = True
    use_classifier: bool = True
    use_structural: bool = True


class SearchPipeline:
    def __init__(self, vi: VersionIndex, cfg: dict, reranker=None):
        self.vi, self.cfg = vi, cfg
        self.dense = DenseRetriever(vi.embedder, vi.vindex, vi.vectors)
        self.bm25 = BM25Retriever(vi.bm25)
        self.hybrid = HybridRetriever(self.dense, self.bm25, vi.ident, cfg)
        self.ranker = Ranker(cfg, vi.chunks, vi.ident, reranker)

    def search(self, query: str, top_k: int = 10, options: PipelineOptions | None = None, category: str | None = None,
               candidate_k: int | None = None) -> tuple[list[SearchResult], dict[str, float]]:
        o = options or PipelineOptions()
        tm: dict[str, float] = {}
        t0 = time.perf_counter()
        q = preprocess_query(query, expand=o.use_expansion and self.cfg["query"].get("expand_concepts", True))
        q.category = category or (classify_query(q) if o.use_classifier and self.cfg["query"].get("classify", True) else "GENERAL_SEMANTIC_SEARCH")
        tm["preprocess_ms"] = (time.perf_counter() - t0) * 1e3
        t1 = time.perf_counter()
        cands = self.hybrid.retrieve(q, candidate_k, o.use_dense, o.use_bm25, o.use_identifier, timings=tm)
        tm["retrieval_ms"] = (time.perf_counter() - t1) * 1e3
        t2 = time.perf_counter()
        results = self.ranker.rank(q, cands, rerank=o.use_rerank, use_structural=o.use_structural and self.cfg["ranking"].get("use_structural", True))
        tm["rerank_ms"] = (time.perf_counter() - t2) * 1e3
        tm["total_ms"] = (time.perf_counter() - t0) * 1e3
        for r in results:
            r.method = r.method  # channels that surfaced it
        top = results[:top_k]
        for r in top:
            r.explanation = f"[{q.category}] " + r.explanation
        return top, tm
