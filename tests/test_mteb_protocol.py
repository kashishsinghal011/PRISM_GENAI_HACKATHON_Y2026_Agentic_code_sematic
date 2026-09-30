"""Exercises our side of the MTEB SearchProtocol with in-memory HF Datasets (no network, no MTEB harness)."""
import pytest

datasets = pytest.importorskip("datasets")
from aci.evaluation.metrics import evaluate_run  # noqa: E402
from aci.evaluation.mteb_model import ACISearchModel  # noqa: E402

DOCS = {
    "d1": "def normalize(s):\n    return s.strip().lower()\n",
    "d2": "def is_locale(s):\n    return s[:5] == 'en-US'\n",
    "d3": "def retry_request(url):\n    for attempt in range(3):\n        try:\n            return get(url)\n        except Exception:\n            sleep(2 ** attempt)\n",
    "d4": "n = int(input())\nprint(sum(range(n)))\n",
}


def test_index_search_contract(cfg):
    corpus = datasets.Dataset.from_dict({"id": list(DOCS), "text": list(DOCS.values()), "title": [""] * len(DOCS)})
    queries = datasets.Dataset.from_dict({"id": ["q1", "q2"], "text": ["strip and lowercase a string", "retry a failed request with backoff"]})
    m = ACISearchModel(cfg)
    with pytest.raises(RuntimeError):
        m.search(queries, top_k=3)
    m.index(corpus)
    run = m.search(queries, top_k=3)
    assert set(run) == {"q1", "q2"} and all(len(v) <= 3 for v in run.values())
    scores = evaluate_run({"q1": {"d1": 1}, "q2": {"d3": 1}}, run)
    assert scores["mrr_at_10"] == 1.0
    assert m.mteb_model_meta.name == "aci/hybrid-code-retrieval"
