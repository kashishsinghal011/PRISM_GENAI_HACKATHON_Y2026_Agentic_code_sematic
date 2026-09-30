"""IR metrics with BEIR/MTEB semantics: nDCG@k (linear gain, log2(rank+1) discount), MRR@k, Recall@k."""
from __future__ import annotations

import math

Qrels = dict[str, dict[str, int]]
Run = dict[str, dict[str, float]]


def _ranked(scores: dict[str, float]) -> list[str]:
    return [d for d, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


def ndcg_at_k(qrels: Qrels, run: Run, k: int = 10) -> float:
    total = 0.0
    for qid, rel in qrels.items():
        ranked = _ranked(run.get(qid, {}))[:k]
        dcg = sum(rel.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked))
        ideal = sorted(rel.values(), reverse=True)[:k]
        idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
        total += dcg / idcg if idcg > 0 else 0.0
    return total / max(1, len(qrels))


def mrr_at_k(qrels: Qrels, run: Run, k: int = 10) -> float:
    total = 0.0
    for qid, rel in qrels.items():
        for i, d in enumerate(_ranked(run.get(qid, {}))[:k]):
            if rel.get(d, 0) > 0:
                total += 1.0 / (i + 1)
                break
    return total / max(1, len(qrels))


def recall_at_k(qrels: Qrels, run: Run, k: int = 100) -> float:
    total = 0.0
    for qid, rel in qrels.items():
        pos = {d for d, r in rel.items() if r > 0}
        if pos:
            total += len(pos & set(_ranked(run.get(qid, {}))[:k])) / len(pos)
    return total / max(1, len(qrels))


def evaluate_run(qrels: Qrels, run: Run) -> dict[str, float]:
    return {"ndcg_at_10": ndcg_at_k(qrels, run, 10), "mrr_at_10": mrr_at_k(qrels, run, 10),
            "recall_at_100": recall_at_k(qrels, run, 100), "recall_at_10": recall_at_k(qrels, run, 10), "n_queries": len(qrels)}
