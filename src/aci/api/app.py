"""Public facade: retrieve(query, repo, version="commit_002")."""
from __future__ import annotations

from ..config import load_config
from ..engine import CodeSearchEngine
from ..indexing.builder import build_index
from ..models import SearchResult


def index_repository(repo_path: str, commit: str | None = None, label: str | None = None, config: str | None = None, **kw):
    return build_index(repo_path, commit=commit, label=label, cfg=load_config(config), **kw)


def retrieve(query: str, repo: str, version: str | None = None, top_k: int = 10, config: str | None = None) -> list[SearchResult]:
    """Ranked code snippets for a natural-language query. `version` may be a label, commit hash, "all" or None (latest)."""
    return CodeSearchEngine(repo, load_config(config)).search(query, version=version, top_k=top_k)
