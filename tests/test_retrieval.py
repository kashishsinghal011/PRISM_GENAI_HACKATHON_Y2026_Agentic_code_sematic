import numpy as np
import pytest
from aci.indexing.store import VersionIndex
from aci.ingestion.chunker import chunk_file, represent
from aci.models import Chunk, SearchResult
from aci.pipeline import PipelineOptions, SearchPipeline
from aci.query.preprocess import preprocess_query
from aci.ranking.dedup import dedup_versions
from aci.ranking.features import FEATURE_NAMES
from aci.retrieval.dense import LSAEmbedder
from aci.retrieval.hybrid import HybridRetriever, _minmax
from aci.retrieval.reranker import LambdaMartReranker, LinearReranker

CODE = {
    "pre.py": "def normalize(text):\n    cleaned = text.strip().lower()\n    return forward(cleaned)\n\ndef forward(v):\n    return main(v)\n\ndef main(v):\n    print(v)\n",
    "chk.py": 'def check(s):\n    pre = s[:6]\n    return pre == "en-US"\n',
    "eng.py": "def perf(s):\n    if act(A, s):\n        return act(B, s)\n",
    "auth.py": "def authenticate_user(u, p):\n    token = jwt.encode({'u': u}, KEY)\n    return token\n\ndef validate_token(t):\n    return jwt.decode(t, KEY)\n",
    "net.py": "def fetch(url):\n    for attempt in range(3):\n        try:\n            return requests.get(url)\n        except Exception:\n            sleep(2 ** attempt)\n",
}


@pytest.fixture
def index(cfg):
    chunks = [c for f, s in CODE.items() for c in chunk_file(s, f, "r", version="v1")]
    for i, c in enumerate(chunks):
        c.doc_id = str(i)
    texts = [represent(c, "contextual") for c in chunks]
    emb = LSAEmbedder(8).fit(texts)
    return VersionIndex.assemble(chunks, emb.encode(texts), emb, cfg)


def test_hybrid_candidates_carry_all_channel_scores(index, cfg):
    p = SearchPipeline(index, cfg)
    q = preprocess_query("authenticate_user jwt token")
    q.category = "FUNCTION_LOOKUP"
    cands = p.hybrid.retrieve(q)
    top = index.chunks[cands[0].idx]
    assert top.function == "authenticate_user" and cands[0].identifier == 1.0
    assert all(0 <= c.dense_n <= 1 and 0 <= c.bm25_n <= 1 for c in cands)
    assert [c.rank for c in cands] == list(range(1, len(cands) + 1))
    assert cands == sorted(cands, key=lambda c: -c.score)


def test_score_fusion_weights_and_normalisation(index, cfg):
    h = SearchPipeline(index, cfg).hybrid
    q = preprocess_query("where is x")
    q.category = "GENERAL_SEMANTIC_SEARCH"
    assert abs(sum(h.weights(q)) - 1) < 1e-9
    assert h.weights(q, use_dense=False)[0] == 0.0
    fl = preprocess_query("authenticate_user"); fl.category = "FUNCTION_LOOKUP"
    assert h.weights(fl)[2] > h.weights(q)[2]
    assert list(_minmax(np.array([1.0, 3.0, 2.0]))) == [0.0, 1.0, 0.5]
    assert list(_minmax(np.array([0.0, 0.0]))) == [0.0, 0.0]


def test_rrf_fusion_mode(index, cfg):
    cfg["retrieval"]["fusion"] = "rrf"
    res, _ = SearchPipeline(index, cfg).search("validate the jwt token", top_k=3)
    assert res and res[0].chunk.function in {"validate_token", "authenticate_user"}


def test_full_pipeline_ranks_expected_snippets_first(index, cfg):
    p = SearchPipeline(index, cfg)
    res, tm = p.search("Where is the input normalized before being passed to the main function?", top_k=3)
    assert res[0].chunk.function == "normalize"
    assert {"preprocess_ms", "total_ms", "rerank_ms"} <= set(tm)
    res, _ = p.search("retry logic for failed requests", top_k=1)
    assert res[0].chunk.function == "fetch"


def test_ablation_switches_change_behaviour(index, cfg):
    p = SearchPipeline(index, cfg)
    r_bm, _ = p.search("jwt decode token", 3, PipelineOptions(use_dense=False, use_identifier=False, use_rerank=False))
    assert all("dense" not in r.method for r in r_bm)
    r_d, _ = p.search("jwt decode token", 3, PipelineOptions(use_bm25=False, use_identifier=False, use_rerank=False))
    assert all("bm25" not in r.method for r in r_d)


def test_structural_feature_rewards_call_graph_neighbours(index, cfg):
    p = SearchPipeline(index, cfg)
    q = preprocess_query("what is passed to the main function"); q.category = "DATA_FLOW"
    cands = p.hybrid.retrieve(q)
    feats, _ = p.ranker.features(q, cands)
    by_fn = {index.chunks[c.idx].function: f for c, f in zip(cands, feats)}
    assert by_fn["forward"]["structural"] == 1.0          # forward() calls main()


def test_linear_reranker_validates_and_reorders():
    with pytest.raises(ValueError):
        LinearReranker({"not_a_feature": 1.0})
    X = np.zeros((3, len(FEATURE_NAMES)), dtype=np.float32)
    X[2, FEATURE_NAMES.index("metadata")] = 1.0
    s = LinearReranker({"metadata": 1.0}).score("q", [], X)
    assert int(np.argmax(s)) == 2


def test_rerank_only_touches_top_n(index, cfg):
    cfg["ranking"]["rerank_top_n"] = 2
    res, _ = SearchPipeline(index, cfg).search("jwt token", top_k=5)
    assert len(res) >= 3 and res[0].score >= res[1].score >= res[2].score


def test_lambdamart_trains_and_scores():
    pytest.importorskip("lightgbm")
    rng = np.random.default_rng(0)
    X = rng.random((60, len(FEATURE_NAMES))).astype(np.float32)
    y = (X[:, 0] > 0.7).astype(int)
    m = LambdaMartReranker.train(X, y, [20, 20, 20], n_estimators=20)
    s = m.score("q", [], X)
    assert s[y == 1].mean() > s[y == 0].mean()
    with pytest.raises(Exception):
        LambdaMartReranker().score("q", [], X)


def _res(chunk, score, vec):
    return SearchResult(chunk=chunk, score=score, vec=np.asarray(vec, dtype=np.float32))


def _ck(fn, ver, file="a.py", code=None):
    return Chunk(repository="r", file=file, language="python", code=code or f"def {fn}(x):\n    return x.strip().lower()", start_line=1, end_line=2,
                 function=fn, version=ver, commit=ver)


def test_dedup_collapses_same_identity_and_renames_but_not_distinct_units():
    v = [1.0, 0.0]
    same = [_res(_ck("f", "v1"), 0.9, v), _res(_ck("f", "v2"), 0.8, v), _res(_ck("f", "v3", code="def f(x):\n    return 1"), 0.7, v)]
    out = dedup_versions(same, [r.vec for r in same], 0.9)
    assert len(out) == 1 and len(out[0].also_in) == 2
    renamed = [_res(_ck("normalize", "v1"), 0.9, v), _res(_ck("sanitize", "v2"), 0.8, v)]
    out = dedup_versions(renamed, [r.vec for r in renamed], 0.9)
    assert len(out) == 1 and "v2:sanitize" in out[0].also_in[0]
    distinct = [_res(_ck("normalize", "v1"), 0.9, v), _res(_ck("other", "v2", code="def other(y):\n    for i in y:\n        yield i"), 0.8, v)]
    assert len(dedup_versions(distinct, [r.vec for r in distinct], 0.9)) == 2
    other_file = [_res(_ck("f", "v1"), 0.9, v), _res(_ck("f", "v2", file="b.py"), 0.8, v)]
    assert len(dedup_versions(other_file, [r.vec for r in other_file], 0.9)) == 2
