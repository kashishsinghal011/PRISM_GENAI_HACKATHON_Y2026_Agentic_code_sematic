"""MTEB (>=2.0) integration.

`ACISearchModel` implements mteb's `SearchProtocol` (index + search + mteb_model_meta), so the *whole*
hybrid pipeline (BM25 + dense + identifier fusion + code-aware rerank), not just an embedder, is what
MTEB scores. Usage:

    task = mteb.get_task("AppsRetrieval")
    result = mteb.evaluate(ACISearchModel(cfg), task)
"""
from __future__ import annotations

import numpy as np

from ..indexing.builder import embed_chunks
from ..indexing.cache import KVCache
from ..indexing.store import VersionIndex
from ..ingestion.chunker import represent
from ..pipeline import PipelineOptions, SearchPipeline
from ..retrieval.dense import load_embedder
from .dataset import doc_to_chunk


def make_model_meta(name: str = "aci/hybrid-code-retrieval", revision: str = "0.1.0"):
    from mteb.models.model_meta import ModelMeta
    return ModelMeta(loader=None, name=name, revision=revision, release_date=None, languages=["eng-Latn", "python-Code"],
                     n_parameters=None, memory_usage_mb=None, max_tokens=None, embed_dim=None, license=None, open_weights=True,
                     public_training_code=None, public_training_data=None, framework=[], similarity_fn_name="cosine",
                     use_instructions=False, training_datasets=set())


class ACISearchModel:
    def __init__(self, cfg: dict, options: PipelineOptions | None = None, name: str = "aci/hybrid-code-retrieval"):
        self.cfg, self.options = cfg, options or PipelineOptions()
        self.pipe: SearchPipeline | None = None
        self.mteb_model_meta = make_model_meta(name)

    # -- SearchProtocol ---------------------------------------------------------------------------------------
    def index(self, corpus, *, task_metadata=None, hf_split=None, hf_subset=None, encode_kwargs=None, num_proc=None) -> None:
        ids = [str(i) for i in corpus["id"]]
        titles = corpus["title"] if "title" in getattr(corpus, "column_names", []) else [""] * len(ids)
        texts = [((t or "") + "\n" + (x or "")).strip() for t, x in zip(titles, corpus["text"])]
        chunks = [doc_to_chunk(i, {"text": t}) for i, t in zip(ids, texts)]
        mode = self.cfg.get("representation", "contextual")
        rep = [represent(c, mode) for c in chunks]
        rng = np.random.default_rng(0)
        fit = [rep[i] for i in rng.choice(len(rep), size=min(len(rep), 60_000), replace=False)]
        embedder = load_embedder(self.cfg, None, fit)
        vecs, _ = embed_chunks(chunks, embedder, self.cfg, KVCache(None), rep)
        self.pipe = SearchPipeline(VersionIndex.assemble(chunks, vecs, embedder, self.cfg), self.cfg)

    def search(self, queries, *, task_metadata=None, hf_split=None, hf_subset=None, top_k: int = 100, encode_kwargs=None,
               top_ranked=None, num_proc=None) -> dict[str, dict[str, float]]:
        if self.pipe is None:
            raise RuntimeError("index() must be called before search().")
        out: dict[str, dict[str, float]] = {}
        for qid, text in zip(queries["id"], queries["text"]):
            if isinstance(text, list):   # conversational queries: use the last user turn
                text = text[-1].get("content", "") if isinstance(text[-1], dict) else str(text[-1])
            try:
                res, _ = self.pipe.search(str(text), top_k=top_k, options=self.options, candidate_k=max(top_k, self.cfg["retrieval"]["candidate_size"]))
            except Exception:
                res = []
            out[str(qid)] = {r.chunk.doc_id: float(r.score) for r in res}
        return out
