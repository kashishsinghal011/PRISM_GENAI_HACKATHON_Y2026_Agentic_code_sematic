"""Build the offline proxy benchmark (docstring -> function) from installed Python source.

  python scripts/make_local_benchmark.py --n-corpus 20000 --n-queries 500 --out data/local_bench
"""
import argparse

import _bootstrap  # noqa: F401
from aci.evaluation.dataset import save_local
from aci.evaluation.local_bench import build_local_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--n-corpus", type=int, default=20000)
ap.add_argument("--n-queries", type=int, default=500)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="data/local_bench")
a = ap.parse_args()
ds = build_local_dataset(a.n_corpus, a.n_queries, a.seed)
save_local(ds, a.out)
print(f"wrote {len(ds.corpus)} docs, {len(ds.queries)} queries to {a.out}")
