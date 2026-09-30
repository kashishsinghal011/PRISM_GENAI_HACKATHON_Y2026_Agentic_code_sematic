"""Latency benchmark at 1K / 10K / 50K / 100K snippets.

  python scripts/benchmark.py --out results/latency.json
"""
import argparse

import _bootstrap  # noqa: F401
from aci.config import load_config
from aci.evaluation.latency import run_latency

ap = argparse.ArgumentParser()
ap.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000, 50000, 100000])
ap.add_argument("--queries", type=int, default=200)
ap.add_argument("--out", default="results/latency.json")
ap.add_argument("--config", default=None)
a = ap.parse_args()
run_latency(load_config(a.config), tuple(a.sizes), a.queries, a.out)
