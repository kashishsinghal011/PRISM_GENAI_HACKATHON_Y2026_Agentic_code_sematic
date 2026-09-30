"""Latency benchmark over growing corpus sizes (proxy corpus of real Python functions)."""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np

from ..indexing.builder import embed_chunks
from ..indexing.store import VersionIndex
from ..ingestion.chunker import represent
from ..models import Chunk
from ..pipeline import SearchPipeline
from ..retrieval.dense import LSAEmbedder
from .local_bench import collect_functions

STAGES = ["preprocess_ms", "query_embed_ms", "ann_ms", "bm25_ms", "rerank_ms", "total_ms"]


def run_latency(cfg: dict, sizes=(1000, 10000, 50000, 100000), n_queries: int = 200, out: str | Path = "results/latency.json", log=print) -> dict:
    need = max(sizes)
    log(f"collecting up to {need} functions ...")
    items = list(collect_functions(limit=need + 2000))
    log(f"collected {len(items)} functions")
    chunks_all = [c for c, _ in items]
    queries = [q for _, q in items if 6 <= len(q.split()) <= 30][:n_queries]
    mode = cfg.get("representation", "contextual")
    results = {"note": "Proxy corpus of installed Python functions; 1 CPU core; LSA fallback encoder.", "queries": len(queries), "sizes": {}}
    for size in sizes:
        if size > len(chunks_all):
            log(f"skipping size {size}: only {len(chunks_all)} functions available")
            results["sizes"][str(size)] = {"skipped": f"only {len(chunks_all)} functions available"}
            continue
        gc.collect()
        chunks = chunks_all[:size]
        t0 = time.perf_counter()
        texts = [represent(c, mode) for c in chunks]
        rng = np.random.default_rng(0)
        fit = [texts[i] for i in rng.choice(len(texts), size=min(len(texts), 30000), replace=False)]
        emb = LSAEmbedder(cfg["embedding_model"].get("lsa_dim", 128)).fit(fit)
        vecs, _ = embed_chunks(chunks, emb, cfg, None, texts)
        vi = VersionIndex.assemble(chunks, vecs, emb, cfg)
        build_s = time.perf_counter() - t0
        pipe = SearchPipeline(vi, cfg)
        for q in queries[:5]:
            pipe.search(q, 10)   # warm-up
        tms = [pipe.search(q, 10)[1] for q in queries]
        row = {"build_seconds": round(build_s, 1), "ann_backend": vi.vindex.backend}
        for s in STAGES:
            vals = np.array([t.get(s, 0.0) for t in tms])
            row[s] = {"mean": round(float(vals.mean()), 2), "p50": round(float(np.percentile(vals, 50)), 2), "p95": round(float(np.percentile(vals, 95)), 2)}
        results["sizes"][str(size)] = row
        log(f"size={size:>6} build={build_s:6.1f}s total mean={row['total_ms']['mean']}ms p95={row['total_ms']['p95']}ms ann={row['ann_ms']['mean']} bm25={row['bm25_ms']['mean']} rerank={row['rerank_ms']['mean']}")
        del vi, pipe, vecs
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(results, indent=1))
    md = ["| Snippets | build (s) | ANN | BM25 | query embed | preprocess | rerank | total mean | total p95 |", "|---|---|---|---|---|---|---|---|---|"]
    for size, r in results["sizes"].items():
        if "skipped" in r:
            md.append(f"| {size} | skipped ({r['skipped']}) |||||||")
        else:
            md.append(f"| {size} | {r['build_seconds']} | {r['ann_ms']['mean']} | {r['bm25_ms']['mean']} | {r['query_embed_ms']['mean']} | {r['preprocess_ms']['mean']} | {r['rerank_ms']['mean']} | {r['total_ms']['mean']} | {r['total_ms']['p95']} |")
    Path(out).with_suffix(".md").write_text("\n".join(md) + "\n\n(all times in ms except build)\n")
    return results
