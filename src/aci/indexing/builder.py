"""Index building with incremental updates.

Flow for a new commit when an earlier version index exists:
    git diff base..commit -> changed files -> reparse only those -> re-embed only new/changed chunks
    (unchanged chunks are copied from the base index and their embeddings come from the cache) -> rebuild
    the (cheap) BM25/ANN structures over the resulting chunk set.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import time
from pathlib import Path, PurePosixPath

import numpy as np

from ..config import load_config
from ..errors import ACIError, IndexNotFoundError
from ..ingestion.chunker import chunk_file, represent
from ..ingestion.parser import EXT_LANG
from ..ingestion.versioning import WORKTREE, Repo
from ..models import Chunk
from ..retrieval.dense import load_embedder
from .cache import EmbeddingCache, KVCache, dumps, loads
from .store import VersionIndex

log = logging.getLogger(__name__)
CHUNKER_VERSION = "1"


def state_dir(cfg: dict, repo_name: str) -> Path:
    return Path(cfg["index"]["dir"]) / repo_name


def read_refs(root: Path) -> dict:
    f = root / "refs.json"
    return json.loads(f.read_text()) if f.exists() else {"versions": {}}


def _write_refs(root: Path, refs: dict) -> None:
    (root / "refs.json").write_text(json.dumps(refs, indent=1))


def _pick_base(root: Path, base: str | None, commit: str, incremental: bool):
    refs = read_refs(root)["versions"]
    if base:
        for lab, v in refs.items():
            if base in (lab, v["commit"]) or v["commit"].startswith(base):
                return lab, v
        raise IndexNotFoundError(f"Base version '{base}' has not been indexed (known: {list(refs) or 'none'}).")
    if not incremental:
        return None
    cands = [(i, lab, v) for i, (lab, v) in enumerate(refs.items()) if v["commit"] != commit]
    if not cands:
        return None
    _, lab, v = max(cands, key=lambda t: (t[2]["timestamp"], t[0]))   # newest commit; ties -> most recently indexed
    return lab, v


def embed_chunks(chunks: list[Chunk], embedder, cfg: dict, kv: KVCache | None, texts: list[str] | None = None) -> tuple[np.ndarray, dict]:
    """Embeddings for chunks, reusing cached vectors (content-addressed keys)."""
    mode = cfg.get("representation", "contextual")
    texts = texts if texts is not None else [represent(c, mode) for c in chunks]
    bs = max(1, int(cfg["embedding_model"].get("batch_size", 32)))
    if getattr(embedder, "kind", "") == "lsa":
        bs = 2048
    vecs = np.zeros((len(chunks), embedder.dim), dtype=np.float32)
    ecache = EmbeddingCache(kv, f"{embedder.model_version}|{mode}", embedder.dim) if kv is not None else None
    keys = [c.cache_key for c in chunks]
    cached = ecache.get_many(keys) if ecache else {}
    todo = [i for i, k in enumerate(keys) if k not in cached]
    for i, k in enumerate(keys):
        if k in cached:
            vecs[i] = cached[k]
    new: dict[str, np.ndarray] = {}
    for s in range(0, len(todo), bs * 8 if bs < 2048 else bs):
        part = todo[s:s + (bs * 8 if bs < 2048 else bs)]
        out = embedder.encode([texts[i] for i in part], batch_size=bs)
        for i, v in zip(part, out):
            vecs[i] = v
            new[keys[i]] = v
    if ecache and new:
        ecache.set_many(new)
    return vecs, {"embeddings_computed": len(todo), "embeddings_from_cache": len(chunks) - len(todo)}


def build_index(repo_path: str, commit: str | None = None, label: str | None = None, cfg: dict | None = None,
                base: str | None = None, incremental: bool = True) -> tuple[VersionIndex, dict]:
    cfg = cfg or load_config()
    t_start = time.perf_counter()
    repo = Repo(repo_path)
    sha = repo.resolve_commit(commit)
    ts = repo.commit_time(sha)
    label = label or ("worktree" if sha == WORKTREE else sha[:8])
    root = state_dir(cfg, repo.name)
    root.mkdir(parents=True, exist_ok=True)
    kv = KVCache(Path(cfg["index"]["cache_dir"]) / f"{repo.name}.sqlite")
    cc = cfg["chunking"]

    files_all = repo.list_files(sha)
    files = {p: v for p, v in files_all.items() if PurePosixPath(p).suffix.lower() in EXT_LANG}
    stats: dict = {"label": label, "commit": sha, "files_total": len(files_all), "files_supported": len(files),
                   "files_unsupported": len(files_all) - len(files), "files_skipped_large": 0, "files_failed": 0}
    if not files:
        raise ACIError(f"No indexable source files in {repo.path} at {sha[:8]} (supported: {sorted(set(EXT_LANG.values()))}).")

    base_sel = _pick_base(root, base, sha, incremental)
    reused: list[Chunk] = []
    to_parse = list(files)
    if base_sel:
        blab, bv = base_sel
        bdir = root / blab
        bman = json.loads((bdir / "manifest.json").read_text())
        bfiles: dict = bman["files"]
        changed = {p for p in files if bfiles.get(p) != files[p][0]}
        if repo.is_git and sha != WORKTREE and bv["commit"] != WORKTREE:
            try:
                git_changed, git_deleted = repo.changed_files(bv["commit"], sha)
                changed |= {p for p in git_changed if p in files}
                stats["git_diff_changed"], stats["git_diff_deleted"] = len(git_changed), len(git_deleted)
            except ACIError as e:
                log.warning("git diff failed (%s); falling back to blob comparison", e)
        for c in VersionIndex.load_chunks(bdir):
            if c.file in files and c.file not in changed:
                reused.append(dataclasses.replace(c, commit=sha, version=label, timestamp=ts))
        to_parse = sorted(changed)
        stats.update(base=blab, files_reparsed=len(to_parse), files_reused=len(files) - len(to_parse))
    else:
        stats.update(base=None, files_reparsed=len(to_parse), files_reused=0)

    # ---- parse (with a per-blob parse cache)
    max_bytes = cc.get("max_file_bytes", 1_000_000)
    pkeys = {p: f"parse|{repo.name}|{files[p][0]}|{cc.get('max_chunk_lines', 60)}|{CHUNKER_VERSION}" for p in to_parse}
    cached = kv.get_many(pkeys.values())
    parsed: list[Chunk] = []
    need_read = []
    for p in to_parse:
        if pkeys[p] in cached:
            parsed += [dataclasses.replace(Chunk.from_dict(d), file=p, commit=sha, version=label, timestamp=ts) for d in loads(cached[pkeys[p]])]
        elif files[p][1] > max_bytes:
            stats["files_skipped_large"] += 1
        else:
            need_read.append(p)
    sources = repo.read_files(sha, files, need_read)
    new_cache = {}
    for p in need_read:
        try:
            ch = chunk_file(sources.get(p, ""), p, repo.name, sha, label, ts, cc.get("max_chunk_lines", 60), max_bytes)
        except Exception as e:  # corrupt / pathological file must not kill the build
            log.warning("Skipping %s: %s", p, e)
            stats["files_failed"] += 1
            continue
        parsed += ch
        new_cache[pkeys[p]] = dumps([c.to_dict() for c in ch])
    if new_cache:
        kv.set_many(new_cache)
    stats["parse_cache_hits"] = len(cached)
    chunks = reused + parsed
    if not chunks:
        raise ACIError("Parsing produced no chunks (all files empty, corrupt or too large).")
    stats.update(chunks_total=len(chunks), chunks_reused=len(reused), chunks_parsed=len(parsed))
    t_parse = time.perf_counter()

    # ---- embed
    mode = cfg.get("representation", "contextual")
    texts = [represent(c, mode) for c in chunks]
    fit_texts = None
    if cfg["embedding_model"].get("backend") != "sentence_transformers" and not (root / "embedder_lsa.pkl").exists():
        rng = np.random.default_rng(0)
        pick = rng.choice(len(texts), size=min(len(texts), 100_000), replace=False)
        fit_texts = [texts[i] for i in pick]
    embedder = load_embedder(cfg, root, fit_texts)
    vecs, est = embed_chunks(chunks, embedder, cfg, kv, texts)
    stats.update(est)
    t_embed = time.perf_counter()

    manifest = {"repository": repo.name, "commit": sha, "label": label, "timestamp": ts, "n_chunks": len(chunks),
                "embedder": embedder.model_version, "representation": mode, "files": {p: v[0] for p, v in files.items()},
                "base": stats.get("base"), "built_at": time.time()}
    vi = VersionIndex.assemble(chunks, vecs, embedder, cfg, manifest)
    vi.save(root / label)
    refs = read_refs(root)
    refs["versions"][label] = {"commit": sha, "timestamp": ts, "n_chunks": len(chunks)}
    _write_refs(root, refs)
    stats.update(seconds_parse=round(t_parse - t_start, 3), seconds_embed=round(t_embed - t_parse, 3),
                 seconds_total=round(time.perf_counter() - t_start, 3), ann_backend=vi.vindex.backend)
    kv.close()
    return vi, stats
