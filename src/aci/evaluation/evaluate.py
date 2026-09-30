"""Evaluation entry points: official MTEB run (needs HuggingFace) and a local/BEIR-style run."""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

from ..pipeline import PipelineOptions, SearchPipeline
from .dataset import RetrievalDataset, build_dataset_index, from_mteb_task, load_local, run_queries
from .metrics import evaluate_run


def run_mteb(cfg: dict, task_name: str = "AppsRetrieval", out_dir: str | Path = "results/mteb", options: PipelineOptions | None = None) -> dict:
    """Runs mteb.evaluate() with our full pipeline as the search model and writes the MTEB result JSON."""
    import mteb  # noqa: WPS433 (heavy optional dependency)
    from .mteb_model import ACISearchModel

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    task = mteb.get_task(task_name)
    model = ACISearchModel(cfg, options)
    t0 = time.time()
    result = mteb.evaluate(model, task, cache=mteb.ResultCache(cache_path=str(out_dir / "mteb_cache")), overwrite_strategy="always")
    wall = time.time() - t0
    (out_dir / f"{task_name}.json").write_text(result.model_dump_json(indent=1) if hasattr(result, "model_dump_json") else json.dumps(str(result)))
    summary = {"task": task_name, "model": model.mteb_model_meta.name, "wall_seconds": round(wall, 1), "platform": platform.platform(),
               "embedding_backend": cfg["embedding_model"]["backend"], "embedding_model": cfg["embedding_model"]["name"]}
    try:
        for tr in result.task_results:
            for split, scores in tr.scores.items():
                for s in scores:
                    summary["scores"] = {k: s[k] for k in ("main_score", "ndcg_at_10", "mrr_at_10", "recall_at_10", "recall_at_100") if k in s}
                    summary["split"] = split
    except Exception as e:  # result layout differs between mteb versions; raw JSON above is authoritative
        summary["parse_note"] = f"could not summarise scores ({e}); see {task_name}.json"
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def run_local(cfg: dict, ds: RetrievalDataset, out_path: str | Path, options: PipelineOptions | None = None) -> dict:
    vi, st = build_dataset_index(ds, cfg)
    run, tms = run_queries(SearchPipeline(vi, cfg), ds.queries, 100, options)
    m = evaluate_run(ds.qrels, run)
    out = {"dataset": ds.name, "n_docs": len(ds.corpus), "metrics": m, "index": st, "note": "local proxy dataset, not AppsRetrieval"}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=1))
    return out
