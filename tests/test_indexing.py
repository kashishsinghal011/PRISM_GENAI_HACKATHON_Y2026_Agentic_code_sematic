import numpy as np
import pytest
from aci.errors import MissingEmbeddingsError, ModelLoadError
from aci.indexing.cache import EmbeddingCache, KVCache
from aci.indexing.lexical_index import BM25Index, IdentifierIndex
from aci.indexing.vector_index import VectorIndex
from aci.ingestion.chunker import chunk_file
from aci.retrieval.dense import LSAEmbedder, load_embedder
from aci.tokenize import tokenize

TEXTS = ["normalize the input string and strip whitespace", "check the language prefix of a locale code", "perform action a then action b",
         "validate the jwt token before database access", "retry the failed http request with exponential backoff"] * 6


def test_tokenizer_keeps_whole_and_split_identifiers():
    t = tokenize("authenticate_user JWTDecoder")
    assert "authenticate_user" in t and "user" in t and "jwtdecoder" in t


def test_lsa_embeddings_shape_norm_determinism():
    e1 = LSAEmbedder(8).fit(TEXTS)
    v = e1.encode(TEXTS[:5])
    assert v.shape == (5, e1.dim) and np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-4)
    assert np.allclose(v, LSAEmbedder(8).fit(TEXTS).encode(TEXTS[:5]), atol=1e-4)
    assert e1.encode(["normalize input"])[0] @ e1.encode([TEXTS[0]])[0] > e1.encode(["normalize input"])[0] @ e1.encode([TEXTS[2]])[0]


def test_unfitted_embedder_errors_and_model_fallback(cfg, tmp_path):
    with pytest.raises(MissingEmbeddingsError):
        LSAEmbedder(8).encode(["x"])
    cfg["embedding_model"].update(backend="sentence_transformers", name="definitely/not-a-model", fallback_to_lsa=True)
    emb = load_embedder(cfg, tmp_path / "st", TEXTS)          # falls back instead of crashing
    assert emb.kind == "lsa"
    cfg["embedding_model"]["fallback_to_lsa"] = False
    with pytest.raises(ModelLoadError):
        load_embedder(cfg, tmp_path / "st2", TEXTS)


def test_embedding_cache_roundtrip_and_dim_guard():
    kv = KVCache(None)
    ec = EmbeddingCache(kv, "m1", 4)
    ec.set_many({"a": np.ones(4, dtype=np.float32)})
    assert set(ec.get_many(["a", "b"])) == {"a"}
    assert EmbeddingCache(kv, "m1", 8).get_many(["a"]) == {}     # different dim -> treated as a miss
    assert EmbeddingCache(kv, "m2", 4).get_many(["a"]) == {}     # different model version -> miss


def test_bm25_rewards_exact_identifiers_and_roundtrips(tmp_path):
    docs = [tokenize(t) for t in ["def authenticate_user(): jwt.decode(token)", "def other(): pass", "render template html"]]
    b = BM25Index().build(docs)
    scores, ids = b.search({"authenticate_user": 2.0, "jwt": 1.0}, 3)
    assert ids[0] == 0 and len(ids) == 1
    b.save(tmp_path / "b.pkl")
    b2 = BM25Index.load(tmp_path / "b.pkl")
    assert np.allclose(b2.score_all({"jwt": 1.0}), b.score_all({"jwt": 1.0}))
    assert b.score_all({"zzz": 1.0}).sum() == 0                  # unknown terms are harmless


def test_identifier_index_defined_vs_called():
    ch = chunk_file("def a():\n    return b()\n\ndef b():\n    return 1\n", "m.py", "r")
    ix = IdentifierIndex().build(ch)
    s = ix.search(["b"])
    assert s[1] == 1.0 and s[0] == 0.5     # b defined in chunk 1, called from chunk 0


@pytest.mark.parametrize("thr,backend", [(10_000, "faiss-flat"), (10, "faiss-hnsw")])
def test_vector_index_returns_nearest(thr, backend, tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 16)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    vi = VectorIndex(16, "hnsw", flat_threshold=thr).build(X)
    assert vi.backend == backend
    s, i = vi.search(X[7], 3)
    assert i[0][0] == 7
    vi.save(tmp_path)
    assert VectorIndex.load(tmp_path, X, flat_threshold=thr).search(X[7], 1)[1][0][0] == 7
    with pytest.raises(ValueError):
        VectorIndex(8).build(X)
