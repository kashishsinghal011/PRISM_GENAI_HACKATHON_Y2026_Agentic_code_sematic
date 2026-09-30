"""Evaluate the retrieval system.

  # Official run (needs internet access to HuggingFace + `pip install mteb datasets`):
  python scripts/evaluate.py --mode mteb --task AppsRetrieval --out results/mteb
  # With a pretrained embedder instead of the CPU fallback:
  python scripts/evaluate.py --mode mteb --set embedding_model.backend=sentence_transformers
  # Offline proxy benchmark (no network):
  python scripts/evaluate.py --mode local --data data/local_bench
"""
import argparse
import json
import sys

import _bootstrap  # noqa: F401
from aci.config import load_config


def parse_sets(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            v = json.loads(v)
        except ValueError:
            pass
        out[k] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["mteb", "local"], default="mteb")
    ap.add_argument("--task", default="AppsRetrieval")
    ap.add_argument("--data", default="data/local_bench")
    ap.add_argument("--out", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", help="config overrides, e.g. retrieval.dense_weight=0.4")
    a = ap.parse_args()
    cfg = load_config(a.config, parse_sets(a.set))
    try:
        if a.mode == "mteb":
            from aci.evaluation.evaluate import run_mteb
            res = run_mteb(cfg, a.task, a.out or "results/mteb")
        else:
            from aci.evaluation.dataset import load_local
            from aci.evaluation.evaluate import run_local
            res = run_local(cfg, load_local(a.data), a.out or "results/local_eval.json")
    except ImportError as e:
        print(f"error: missing optional dependency for --mode {a.mode}: {e}\nInstall with: pip install mteb datasets", file=sys.stderr)
        return 2
    except Exception as e:  # network/dataset errors are the common failure here
        print(f"error: evaluation failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
