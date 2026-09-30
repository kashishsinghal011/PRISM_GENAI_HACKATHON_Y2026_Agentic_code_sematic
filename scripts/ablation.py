"""Ablation study on the offline proxy benchmark (see README: this is NOT AppsRetrieval).

  python scripts/ablation.py --data data/local_bench --out results
"""
import argparse

import _bootstrap  # noqa: F401
from aci.config import load_config
from aci.evaluation.ablation import run_ablation
from aci.evaluation.dataset import load_local

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/local_bench")
ap.add_argument("--out", default="results")
ap.add_argument("--trials", type=int, default=30)
ap.add_argument("--config", default=None)
a = ap.parse_args()
run_ablation(load_local(a.data), load_config(a.config), a.out, a.trials)
