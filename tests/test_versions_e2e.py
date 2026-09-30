import json

import pytest
from aci.engine import CodeSearchEngine
from aci.errors import (EmptyQueryError, IndexNotFoundError, InvalidCommitError, MissingEmbeddingsError,
                        RepositoryNotFoundError, ACIError)
from aci.indexing.builder import build_index
from aci.ingestion.versioning import Repo


def build(repo, commits, cfg, **kw):
    out = []
    for n, c in enumerate(commits, 1):
        out.append(build_index(str(repo), c, f"commit_00{n}", cfg, **kw)[1])
    return out


def test_end_to_end_repo_index_query_results(git_repo, cfg):
    repo, commits = git_repo
    build(repo, commits, cfg)
    eng = CodeSearchEngine(repo, cfg)
    res = eng.search("Where is the input normalized before being passed to the main function?", version="commit_003", top_k=3)
    assert res[0].chunk.function == "sanitize_input" and res[0].chunk.file == "preprocessing.py"
    assert res[0].chunk.start_line == 1 and res[0].chunk.commit == commits[2] and res[0].chunk.version == "commit_003"
    res = eng.search("Where is the JWT token validated?", version="commit_002", top_k=2)
    assert res[0].chunk.function == "validate_token"
    assert eng.search("validate jwt token", version=commits[1][:8], top_k=1)      # commit-hash prefix resolves


def test_version_selection_returns_that_versions_code(git_repo, cfg):
    repo, commits = git_repo
    build(repo, commits, cfg)
    eng = CodeSearchEngine(repo, cfg)
    names = {v: eng.search("normalize input", version=v, top_k=1)[0].chunk.function for v in ("commit_001", "commit_002", "commit_003")}
    assert names == {"commit_001": "normalize", "commit_002": "preprocess_input", "commit_003": "sanitize_input"}


def test_incremental_indexing_reuses_unchanged_work(git_repo, cfg):
    repo, commits = git_repo
    s1, s2, s3 = build(repo, commits, cfg)
    assert s1["base"] is None and s1["embeddings_computed"] == s1["chunks_total"]
    assert s2["base"] == "commit_001" and s2["git_diff_changed"] == 2
    assert s2["files_reused"] == 1 and s2["chunks_reused"] >= 1                     # engine.py untouched
    assert s2["embeddings_computed"] == 2 and s2["embeddings_from_cache"] == s2["chunks_total"] - 2   # renamed fn + new fn only
    assert s3["embeddings_computed"] == 1                                            # only sanitize_input is new
    full = build_index(str(repo), commits[2], "rebuild", cfg, incremental=False)[1]  # full rebuild still hits the embedding cache
    assert full["embeddings_computed"] == 0 and full["base"] is None


def test_incremental_result_equals_full_rebuild(git_repo, cfg):
    repo, commits = git_repo
    build(repo, commits, cfg)
    a, _ = build_index(str(repo), commits[2], "inc", cfg, base="commit_002")
    b, _ = build_index(str(repo), commits[2], "full", cfg, incremental=False)
    key = lambda vi: sorted((c.file, c.qualname, c.start_line, c.chash) for c in vi.chunks)
    assert key(a) == key(b)


def test_cross_version_search_dedups_unless_history_requested(git_repo, cfg):
    repo, commits = git_repo
    build(repo, commits, cfg)
    eng = CodeSearchEngine(repo, cfg)
    collapsed = eng.search("normalize input", version="all", top_k=10)
    full = eng.search("normalize input", version="all", top_k=30, history=True)
    assert len(collapsed) < len(full)
    top = collapsed[0]
    assert top.chunk.file == "preprocessing.py" and len(top.also_in) == 2      # all 3 renames folded into one result


def test_error_handling(git_repo, cfg, tmp_path):
    repo, commits = git_repo
    with pytest.raises(RepositoryNotFoundError):
        Repo(tmp_path / "missing")
    with pytest.raises(InvalidCommitError):
        build_index(str(repo), "deadbeefdeadbeef", "x", cfg)
    with pytest.raises(IndexNotFoundError):
        CodeSearchEngine("never_indexed", cfg)
    build_index(str(repo), commits[0], "commit_001", cfg)
    eng = CodeSearchEngine(repo, cfg)
    with pytest.raises(EmptyQueryError):
        eng.search("   ")
    with pytest.raises(IndexNotFoundError):
        eng.search("x", version="nope")
    with pytest.raises(ValueError):
        eng.search("x", top_k=0)
    with pytest.raises(IndexNotFoundError):
        build_index(str(repo), commits[1], "c2", cfg, base="not_a_version")


def test_missing_and_mismatched_artifacts(git_repo, cfg):
    repo, commits = git_repo
    build_index(str(repo), commits[0], "commit_001", cfg)
    root = next((p for p in (__import__("pathlib").Path(cfg["index"]["dir"])).iterdir()))
    (root / "commit_001" / "vectors.npy").unlink()
    with pytest.raises(IndexNotFoundError):
        CodeSearchEngine(repo, cfg).search("x y")


def test_plain_directory_and_corrupt_and_unsupported_files(tmp_path, cfg):
    d = tmp_path / "plain"
    d.mkdir()
    (d / "ok.py").write_text("def hello(name):\n    return 'hi ' + name\n")
    (d / "broken.py").write_text("def oops(:\n   ===\n")
    (d / "readme.md").write_text("# not code")
    (d / "huge.py").write_text("x = 1\n" * 10)
    cfg["chunking"]["max_file_bytes"] = 50
    _, st = build_index(str(d), None, None, cfg)
    assert st["label"] == "worktree" and st["files_unsupported"] == 1
    empty = tmp_path / "empty"; empty.mkdir(); (empty / "a.txt").write_text("x")
    with pytest.raises(ACIError):
        build_index(str(empty), None, None, cfg)
    with pytest.raises(InvalidCommitError):
        build_index(str(d), "abc123", None, cfg)


def test_metrics_match_reference():
    from aci.evaluation.metrics import mrr_at_k, ndcg_at_k
    qrels = {"q1": {"a": 1}, "q2": {"b": 1, "c": 1}}
    run = {"q1": {"x": 3, "a": 2, "y": 1}, "q2": {"c": 5, "z": 4, "b": 3}}
    assert abs(mrr_at_k(qrels, run) - (0.5 + 1.0) / 2) < 1e-9
    import math
    n1 = (1 / math.log2(3)) / 1.0
    n2 = (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert abs(ndcg_at_k(qrels, run) - (n1 + n2) / 2) < 1e-9
    try:
        import pytrec_eval
    except ImportError:
        return
    ev = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10", "recip_rank"})
    out = ev.evaluate(run)
    assert abs(sum(v["ndcg_cut_10"] for v in out.values()) / 2 - ndcg_at_k(qrels, run)) < 1e-6
