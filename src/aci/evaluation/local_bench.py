"""Local docstring->function benchmark built from installed Python source (no network needed).

Why: the real CoIR AppsRetrieval data lives on HuggingFace, which this sandbox cannot reach. This proxy
gives *measured* (never invented) numbers for ablations/latency, but it is NOT AppsRetrieval and its
absolute scores must not be compared with CoIR/MTEB leaderboards.

Construction: every function/method (3-60 lines) becomes a corpus doc with its docstring REMOVED; the
first sentence of its docstring is the query; the function itself is the single relevant document.
"""
from __future__ import annotations

import ast
import random
import re
import sys
import sysconfig
import textwrap
from pathlib import Path

from ..ingestion.parser import parse_python
from ..models import Chunk
from .dataset import RetrievalDataset


def default_roots() -> list[str]:
    roots = {sysconfig.get_paths()[k] for k in ("stdlib", "purelib", "platlib")}
    return sorted(r for r in roots if Path(r).is_dir())


def strip_docstring(code: str) -> str:
    try:
        fn = ast.parse(textwrap.dedent(code)).body[0]
    except Exception:
        return code
    if fn.body and isinstance(fn.body[0], ast.Expr) and isinstance(getattr(fn.body[0], "value", None), ast.Constant) and isinstance(fn.body[0].value.value, str):
        lines = code.splitlines()
        del lines[fn.body[0].lineno - 1: fn.body[0].end_lineno]
        return "\n".join(lines)
    return code


def first_sentence(doc: str) -> str:
    para = " ".join(doc.strip().split("\n\n")[0].split())
    m = re.search(r"(?<=[a-z0-9\)])\.\s", para + " ")
    return (para[: m.start() + 1] if m else para).strip()


def collect_functions(roots: list[str] | None = None, limit: int | None = None, seed: int = 0, max_file_bytes: int = 200_000):
    """Yield (chunk_without_docstring, first_sentence_of_docstring) for functions/methods found under roots."""
    files = [p for r in (roots or default_roots()) for p in Path(r).rglob("*.py")]
    random.Random(seed).shuffle(files)
    seen: set[str] = set()
    n = 0
    for p in files:
        try:
            if p.stat().st_size > max_file_bytes:
                continue
            src = p.read_text(encoding="utf-8", errors="replace")
            chunks = parse_python(src, str(p), dict(repository="pylib", file=str(p.name), language="python", commit="", version="", timestamp=0.0), 60)
        except Exception:
            continue
        for c in chunks:
            if c.kind not in ("function", "method") or not (3 <= c.end_line - c.start_line + 1 <= 60):
                continue
            body = strip_docstring(c.code)
            key = " ".join(body.split())
            if key in seen or len(body) < 40:
                continue
            seen.add(key)
            q = first_sentence(c.doc) if c.doc else ""
            c.code, c.doc, c.chash = body, "", ""
            c.__post_init__()
            c.file = "/".join(p.parts[-3:])   # short, human-readable path
            yield c, q
            n += 1
            if limit and n >= limit:
                return


def build_local_dataset(n_corpus: int = 20_000, n_queries: int = 500, seed: int = 0, roots: list[str] | None = None) -> RetrievalDataset:
    items = list(collect_functions(roots, limit=n_corpus, seed=seed))
    corpus, queries, qrels = {}, {}, {}
    cand = []
    seen_q: set[str] = set()
    for i, (c, q) in enumerate(items):
        did = f"d{i}"
        c.doc_id = did
        corpus[did] = {"text": c.code, "chunk": c.to_dict()}
        w = len(q.split())
        if 6 <= w <= 30 and q.lower() not in seen_q and not q.lower().startswith(("return self", "see ")):
            seen_q.add(q.lower())
            cand.append((did, q))
    rng = random.Random(seed)
    for did, q in rng.sample(cand, min(n_queries, len(cand))):
        qid = f"q{len(queries)}"
        queries[qid], qrels[qid] = q, {did: 1}
    return RetrievalDataset(corpus, queries, qrels, name="local_pylib")
