"""Ablation study + weight tuning + LambdaMART + version experiment. Every number is measured, never estimated."""
from __future__ import annotations

import copy
import json
import re
import time
from pathlib import Path

import numpy as np

from ..indexing.cache import KVCache
from ..indexing.store import VersionIndex
from ..ingestion.chunker import represent
from ..models import Chunk
from ..pipeline import PipelineOptions, SearchPipeline
from ..ranking.dedup import dedup_versions
from ..ranking.features import FEATURE_NAMES
from ..ranking.ranker import Ranker
from ..retrieval.reranker import LambdaMartReranker
from .dataset import RetrievalDataset, build_dataset_index, doc_to_chunk, run_queries
from .metrics import evaluate_run, mrr_at_k

BM25_ONLY = PipelineOptions(use_dense=False, use_identifier=False, use_rerank=False, use_expansion=False, use_classifier=False, use_structural=False)
DENSE_ONLY = PipelineOptions(use_bm25=False, use_identifier=False, use_rerank=False, use_expansion=False, use_classifier=False, use_structural=False)
HYBRID = PipelineOptions(use_identifier=False, use_rerank=False, use_expansion=False, use_classifier=False, use_structural=False)
HYBRID_ID = PipelineOptions(use_rerank=False, use_expansion=False, use_classifier=False, use_structural=False)
RERANK_BASIC = PipelineOptions(use_expansion=False, use_classifier=False, use_structural=False)
PLUS_QUERY = PipelineOptions(use_structural=False)
FULL = PipelineOptions()

W_BASIC = {"semantic": 0.6, "bm25": 0.25, "identifier": 0.15}
W_META = {"semantic": 0.5, "bm25": 0.2, "identifier": 0.1, "metadata": 0.2}


def split_queries(ds: RetrievalDataset, seed: int = 0, val_frac: float = 0.5):
    ids = sorted(ds.queries)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_val = int(len(ids) * val_frac)
    pick = lambda sel: ({q: ds.queries[q] for q in sel}, {q: ds.qrels[q] for q in sel})
    return pick(ids[:n_val]), pick(ids[n_val:])


def with_cfg(cfg: dict, **over) -> dict:
    c = copy.deepcopy(cfg)
    for k, v in over.items():
        sec, key = k.split("__")
        c[sec][key] = v
    return c


def measure(pipe: SearchPipeline, queries, qrels, options, top_k=100) -> dict:
    t0 = time.perf_counter()
    run, tms = run_queries(pipe, queries, top_k, options)
    m = evaluate_run(qrels, run)
    m["mean_latency_ms"] = round(float(np.mean([t.get("total_ms", 0) for t in tms])), 2)
    m["wall_s"] = round(time.perf_counter() - t0, 1)
    return {k: (round(v, 4) if isinstance(v, float) and k != "mean_latency_ms" else v) for k, v in m.items()}


def tune_weights(vi, cfg, val_q, val_r, n_trials=30, seed=0) -> tuple[dict, list[dict]]:
    rng = np.random.default_rng(seed)
    trials = []
    base = {"retrieval": {k: cfg["retrieval"][k] for k in ("dense_weight", "bm25_weight", "identifier_weight")}, "weights": dict(cfg["ranking"]["weights"])}
    cands = [base]
    for _ in range(n_trials):
        rw = rng.dirichlet([2.0, 1.5, 0.7])
        w = rng.dirichlet([3, 1.5, 0.7, 1, 0.7])
        cands.append({"retrieval": dict(zip(("dense_weight", "bm25_weight", "identifier_weight"), map(float, rw))),
                      "weights": dict(zip(("semantic", "bm25", "identifier", "metadata", "structural"), map(float, w)))})
    best, best_score = base, -1.0
    for c in cands:
        cc = copy.deepcopy(cfg)
        cc["retrieval"].update(c["retrieval"])
        cc["ranking"]["weights"] = c["weights"]
        m = measure(SearchPipeline(vi, cc), val_q, val_r, FULL, top_k=10)
        trials.append({**c, "val_ndcg_at_10": m["ndcg_at_10"], "val_mrr_at_10": m["mrr_at_10"]})
        if m["ndcg_at_10"] > best_score:
            best, best_score = c, m["ndcg_at_10"]
    return best, trials


def train_lambdamart(pipe: SearchPipeline, val_q, val_r) -> LambdaMartReranker | None:
    from ..query.classifier import classify_query
    from ..query.preprocess import preprocess_query
    X, y, groups = [], [], []
    for qid, text in val_q.items():
        q = preprocess_query(text)
        q.category = classify_query(q)
        cands = pipe.hybrid.retrieve(q)
        if not cands:
            continue
        _, F = pipe.ranker.features(q, cands)
        rel = val_r[qid]
        X.append(F)
        y += [1 if pipe.vi.chunks[c.idx].doc_id in rel else 0 for c in cands]
        groups.append(len(cands))
    if not X or sum(y) == 0:
        return None
    return LambdaMartReranker.train(np.vstack(X), np.array(y), groups)


def refactor(chunk: Chunk, version: str) -> Chunk:
    """Simulate a rename-refactor: new function name + a trailing comment (different content hash and identity)."""
    c = Chunk.from_dict(chunk.to_dict())
    c.version = version
    if c.function:
        new = f"refactored_{c.function}"
        c.code = re.sub(rf"\bdef {re.escape(c.function)}\b", f"def {new}", c.code, count=1) + "\n# refactored"
        c.signature = c.signature.replace(c.function, new, 1)
        c.function = new
    c.chash = ""
    c.__post_init__()
    return c


def version_experiment(ds: RetrievalDataset, cfg: dict, n_units: int = 4000, edit_frac: float = 0.3, n_queries: int = 200, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    ids = list(ds.corpus)[:n_units]
    idset = set(ids)
    v1 = [doc_to_chunk(i, ds.corpus[i]) for i in ids]
    for c in v1:
        c.version, c.commit = "v1", "v1"
    edited = set(rng.choice(len(v1), size=int(len(v1) * edit_frac), replace=False).tolist())
    v2 = [refactor(c, "v2") if i in edited else Chunk.from_dict({**c.to_dict(), "version": "v2", "commit": "v2"}) for i, c in enumerate(v1)]
    mode = cfg.get("representation", "contextual")
    kv = KVCache(None)
    from ..indexing.builder import embed_chunks
    from ..retrieval.dense import LSAEmbedder
    emb = LSAEmbedder(cfg["embedding_model"].get("lsa_dim", 128)).fit([represent(c, mode) for c in v1])
    e1, s1 = embed_chunks(v1, emb, cfg, kv)
    e2, s2 = embed_chunks(v2, emb, cfg, kv)
    pipes = [SearchPipeline(VersionIndex.assemble(cs, e, emb, cfg), cfg) for cs, e in ((v1, e1), (v2, e2))]
    qs = [(q, t) for q, t in ds.queries.items() if next(iter(ds.qrels[q])) in idset][:n_queries]
    out = {"n_units": len(ids), "edited_fraction": edit_frac, "embeddings_v1": s1, "embeddings_v2_incremental": s2}
    stats = {"no_dedup": {"rr": [], "dup": []}, "dedup": {"rr": [], "dup": []}}
    for qid, text in qs:
        target = next(iter(ds.qrels[qid]))
        merged = []
        for p in pipes:
            res, _ = p.search(text, top_k=30)
            for r in res:
                r.vec = p.vi.vectors[p.vi.chunks.index(r.chunk)] if False else None
            merged += res
        pos = {}
        for p in pipes:
            pos[p.vi.manifest.get("label", id(p))] = {id(c): i for i, c in enumerate(p.vi.chunks)}
        for r in merged:
            for p in pipes:
                i = pos[p.vi.manifest.get("label", id(p))].get(id(r.chunk))
                if i is not None:
                    r.vec = p.vi.vectors[i]
        merged.sort(key=lambda r: -r.score)
        for name, lst in (("no_dedup", merged), ("dedup", dedup_versions(merged, [r.vec for r in merged], cfg["dedup"]["similarity_threshold"]))):
            top = [r.chunk.doc_id for r in lst[:10]]
            seen, dups = set(), 0
            for d in top:
                dups += d in seen
                seen.add(d)
            stats[name]["dup"].append(dups / 10)
            stats[name]["rr"].append(next((1 / (i + 1) for i, d in enumerate(top) if d == target), 0.0))
    for k, v in stats.items():
        out[k] = {"unit_mrr_at_10": round(float(np.mean(v["rr"])), 4), "duplicate_rate_at_10": round(float(np.mean(v["dup"])), 4)}
    out["n_queries"] = len(qs)
    return out


def run_ablation(ds: RetrievalDataset, cfg: dict, out_dir: str | Path, n_trials: int = 30, log=print) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (val_q, val_r), (test_q, test_r) = split_queries(ds)
    log(f"corpus={len(ds.corpus)} val_queries={len(val_q)} test_queries={len(test_q)}")
    rows: list[dict] = []

    def add(name, m, note=""):
        rows.append({"approach": name, **m, "note": note})
        log(f"{name:<44} MRR@10={m['mrr_at_10']:.4f} NDCG@10={m['ndcg_at_10']:.4f} R@100={m['recall_at_100']:.4f} lat={m['mean_latency_ms']}ms")

    indexes = {}
    for mode in ("contextual", "raw", "metadata"):
        c = with_cfg(cfg, **{"__".join(("representation",)): mode}) if False else copy.deepcopy(cfg)
        c["representation"] = mode
        vi, st = build_dataset_index(ds, c)
        indexes[mode] = (vi, c)
        log(f"built {mode} index: {st}")
    vi, c0 = indexes["contextual"]
    P = lambda over=None, vi_=vi, cfg_=c0: SearchPipeline(vi_, cfg_ if over is None else with_cfg(cfg_, **over))
    add("1. BM25 only", measure(P(), test_q, test_r, BM25_ONLY))
    add("2. Dense only (LSA)", measure(P(), test_q, test_r, DENSE_ONLY))
    add("3. Dense + BM25 (hybrid)", measure(P(), test_q, test_r, HYBRID))
    add("4. Hybrid + identifier channel", measure(P(), test_q, test_r, HYBRID_ID))
    cb = copy.deepcopy(c0); cb["ranking"]["weights"] = W_BASIC
    add("5. Hybrid + reranker (basic features)", measure(SearchPipeline(vi, cb), test_q, test_r, RERANK_BASIC))
    add("6. + query preprocessing", measure(SearchPipeline(vi, cb), test_q, test_r, PLUS_QUERY))
    cm = copy.deepcopy(c0); cm["ranking"]["weights"] = W_META
    add("7. + metadata features", measure(SearchPipeline(vi, cm), test_q, test_r, PLUS_QUERY))
    add("8. + structural features (Full, default weights)", measure(P(), test_q, test_r, FULL))
    best, trials = tune_weights(vi, c0, val_q, val_r, n_trials)
    ct = copy.deepcopy(c0); ct["retrieval"].update(best["retrieval"]); ct["ranking"]["weights"] = best["weights"]
    add("9. Full system, weights tuned on val", measure(SearchPipeline(vi, ct), test_q, test_r, FULL), json.dumps({k: {a: round(b, 3) for a, b in v.items()} for k, v in best.items()}))
    lm = train_lambdamart(SearchPipeline(vi, c0), val_q, val_r)
    if lm is not None:
        pl = SearchPipeline(vi, c0, reranker=lm)
        add("10. Full features + LambdaMART (trained on val)", measure(pl, test_q, test_r, FULL))
    # code-representation experiment (which text is embedded / indexed)
    rep_rows = []
    for mode in ("raw", "metadata", "contextual"):
        vi_m, c_m = indexes[mode]
        for label, opt in (("dense only", DENSE_ONLY), ("hybrid", HYBRID)):
            m = measure(SearchPipeline(vi_m, c_m), test_q, test_r, opt)
            rep_rows.append({"representation": mode, "system": label, **m})
            log(f"representation={mode:<10} {label:<10} MRR@10={m['mrr_at_10']:.4f} NDCG@10={m['ndcg_at_10']:.4f}")
    ver = version_experiment(ds, c0)
    log(f"version experiment: {ver}")
    result = {"dataset": ds.name, "n_corpus": len(ds.corpus), "n_test_queries": len(test_q), "n_val_queries": len(val_q),
              "note": "Proxy benchmark (docstring->function over installed Python source), NOT CoIR AppsRetrieval.",
              "ablation": rows, "representation": rep_rows, "version_experiment": ver, "tuning_trials": trials}
    (out_dir / "ablation_local.json").write_text(json.dumps(result, indent=1))
    md = ["| Approach | MRR@10 | NDCG@10 | Recall@100 | mean latency (ms) |", "|---|---|---|---|---|"]
    md += [f"| {r['approach']} | {r['mrr_at_10']:.4f} | {r['ndcg_at_10']:.4f} | {r['recall_at_100']:.4f} | {r['mean_latency_ms']} |" for r in rows]
    md += ["", "| Representation | System | MRR@10 | NDCG@10 |", "|---|---|---|---|"]
    md += [f"| {r['representation']} | {r['system']} | {r['mrr_at_10']:.4f} | {r['ndcg_at_10']:.4f} |" for r in rep_rows]
    (out_dir / "ablation_local.md").write_text("\n".join(md) + "\n")
    return result
