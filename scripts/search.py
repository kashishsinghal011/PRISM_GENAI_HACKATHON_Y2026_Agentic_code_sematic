"""Search an indexed repository.

  python scripts/search.py --repo myrepo --query "Where is the input normalized before main?" --top-k 10
  python scripts/search.py --repo myrepo --query "retry logic" --version all
"""
import argparse
import json
import sys

import _bootstrap  # noqa: F401
from aci.config import load_config
from aci.engine import CodeSearchEngine
from aci.errors import ACIError


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="repository path or indexed repository name")
    ap.add_argument("--query", required=True)
    ap.add_argument("--version", default=None, help="label/commit, or 'all' (default: latest)")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--history", action="store_true", help="with --version all: keep every version instead of collapsing")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    try:
        eng = CodeSearchEngine(a.repo, load_config(a.config))
        res = eng.search(a.query, version=a.version, top_k=a.top_k, history=a.history)
    except (ACIError, ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if a.json:
        print(json.dumps([{"rank": r.rank, "score": round(r.score, 4), "file": r.chunk.file, "function": r.chunk.qualname, "start_line": r.chunk.start_line,
                           "end_line": r.chunk.end_line, "version": r.chunk.version, "method": r.method, "why": r.explanation, "also_in": r.also_in} for r in res], indent=1))
        return 0
    for r in res:
        c = r.chunk
        print(f"{r.rank:>2}. {c.file}  {c.qualname}  L{c.start_line}-{c.end_line}  [{c.version}]  score={r.score:.3f}  via {r.method}")
        print(f"    {r.explanation}" + (f"  (also in: {', '.join(r.also_in)})" if r.also_in else ""))
    print(f"\n{len(res)} result(s) in {eng.last_timings.get('total_ms', 0):.1f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
