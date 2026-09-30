"""BEIR-style retrieval datasets: local (JSONL) and MTEB tasks, plus index construction from them."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from ..indexing.builder import embed_chunks
from ..indexing.cache import KVCache
from ..indexing.store import VersionIndex
from ..ingestion.chunker import represent
from ..ingestion.parser import parse_source
from ..models import Chunk
from ..pipeline import PipelineOptions, SearchPipeline
from ..retrieval.dense import LSAEmbedder, load_embedder


@dataclass
class RetrievalDataset:
    corpus: dict[str, dict]                # doc id -> {"text": str, "chunk": Chunk-dict | None}
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]
    name: str = "dataset"


def save_local(ds: RetrievalDataset, d: str | Path) -> None:
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "corpus.jsonl", "w") as f:
        for i, doc in ds.corpus.items():
            f.write(json.dumps({"id": i, **doc}) + "\n")
    with open(d / "queries.jsonl", "w") as f:
        for i, t in ds.queries.items():
            f.write(json.dumps({"id": i, "text": t}) + "\n")
    (d / "qrels.json").write_text(json.dumps(ds.qrels))


def load_local(d: str | Path) -> RetrievalDataset:
    d = Path(d)
    if not (d / "corpus.jsonl").exists():
        raise FileNotFoundError(f"Local benchmark not found at {d}. Create it: python scripts/make_local_benchmark.py")
    corpus = {}
    for l in open(d / "corpus.jsonl"):
        j = json.loads(l)
        corpus[j.pop("id")] = j
    queries = {j["id"]: j["text"] for j in map(json.loads, open(d / "queries.jsonl"))}
    return RetrievalDataset(corpus, queries, json.loads((d / "qrels.json").read_text()), name=d.name)


def from_mteb_task(task, split: str | None = None) -> RetrievalDataset:
    """Materialise an MTEB retrieval task (e.g. mteb.get_task("AppsRetrieval")). Needs HuggingFace access."""
    task.load_data()
    split = split or task.metadata.eval_splits[0]
    corpus_ds, queries_ds, rel = task.corpus[split], task.queries[split], task.relevant_docs[split]
    corpus = {str(i): {"text": ((t or "") + "\n" + (x or "")).strip() if (t := (title or "")) else (x or ""), "chunk": None}
              for i, title, x in zip(corpus_ds["id"], corpus_ds["title"] if "title" in corpus_ds.column_names else [""] * len(corpus_ds), corpus_ds["text"])}
    queries = {str(i): t for i, t in zip(queries_ds["id"], queries_ds["text"])}
    qrels = {str(q): {str(d): int(s) for d, s in docs.items()} for q, docs in rel.items()}
    return RetrievalDataset(corpus, queries, qrels, name=task.metadata.name)


def doc_to_chunk(doc_id: str, doc: dict) -> Chunk:
    """One retrieval unit per corpus document. Keeps parsed metadata (function name, calls, imports) when the text parses."""
    if doc.get("chunk"):
        c = Chunk.from_dict(doc["chunk"])
        c.doc_id = doc_id
        return c
    text = doc["text"]
    base = dict(repository="corpus", commit="", version="", timestamp=0.0)
    first = None
    try:
        parts = parse_source(text, f"{doc_id}.py", base, max_lines=10**6)
        first = next((p for p in parts if p.function), None)
        calls = list(dict.fromkeys(x for p in parts for x in p.calls))
        imports = parts[0].imports if parts else []
    except Exception:
        calls, imports = [], []
    return Chunk(repository="corpus", file=f"{doc_id}.py", language="python", code=text, start_line=1, end_line=text.count("\n") + 1,
                 function=first.function if first else "", signature=first.signature if first else "", doc=first.doc if first else "",
                 imports=imports, calls=calls, kind="file", doc_id=doc_id)


def build_dataset_index(ds: RetrievalDataset, cfg: dict, embedder=None, kv: KVCache | None = None, max_fit: int = 60_000):
    chunks = [doc_to_chunk(i, d) for i, d in ds.corpus.items()]
    mode = cfg.get("representation", "contextual")
    texts = [represent(c, mode) for c in chunks]
    t0 = time.perf_counter()
    if embedder is None:
        rng = np.random.default_rng(0)
        pick = rng.choice(len(texts), size=min(len(texts), max_fit), replace=False)
        embedder = load_embedder(cfg, None, [texts[i] for i in pick])
    vecs, st = embed_chunks(chunks, embedder, cfg, kv, texts)
    vi = VersionIndex.assemble(chunks, vecs, embedder, cfg)
    return vi, {"build_seconds": round(time.perf_counter() - t0, 2), **st, "n_docs": len(chunks), "ann_backend": vi.vindex.backend}


def run_queries(pipe: SearchPipeline, queries: dict[str, str], top_k: int = 100, options: PipelineOptions | None = None,
                progress: Callable[[int], None] | None = None) -> tuple[dict[str, dict[str, float]], list[dict]]:
    run: dict[str, dict[str, float]] = {}
    timings: list[dict] = []
    for n, (qid, text) in enumerate(queries.items()):
        try:
            res, tm = pipe.search(text, top_k=top_k, options=options, candidate_k=max(top_k, pipe.cfg["retrieval"]["candidate_size"]))
        except Exception:  # an unparseable/empty query must not abort a whole evaluation
            res, tm = [], {}
        run[qid] = {r.chunk.doc_id: r.score for r in res}
        timings.append(tm)
        if progress and n % 100 == 0:
            progress(n)
    return run, timings
