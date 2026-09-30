"""Index the bundled sample repo (examples/sample_repo) and run the spec's demo queries.

  python scripts/demo_queries.py            # writes results/demo_queries.md
"""
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from aci.config import load_config
from aci.engine import CodeSearchEngine
from aci.indexing.builder import build_index

QUERIES = [
    "Where is the user input sanitized before being passed to the API?",
    "Where is authentication performed before accessing the database?",
    "Which function converts the incoming request into the internal representation?",
    "Where is the database connection initialized?",
    "Which code handles retry logic for failed API calls?",
    "Where is the JWT token validated?",
    "Where is the input normalized before reaching the main function?",
]

root = Path(__file__).resolve().parents[1]
repo = root / "examples" / "sample_repo"
cfg = load_config()
_, st = build_index(str(repo), None, "worktree", cfg, incremental=False)
eng = CodeSearchEngine(repo, cfg)
lines = [f"# Demo queries on `examples/sample_repo` ({st['chunks_total']} chunks, encoder={cfg['embedding_model']['backend']})", ""]
for q in QUERIES:
    res = eng.search(q, top_k=3)
    lines += [f"## {q}", ""]
    for r in res:
        c = r.chunk
        lines.append(f"{r.rank}. `{c.file}` `{c.qualname}` L{c.start_line}-{c.end_line} score={r.score:.3f} via {r.method}: {r.explanation}")
    lines.append("")
out = root / "results" / "demo_queries.md"
out.write_text("\n".join(lines))
print("\n".join(lines))
