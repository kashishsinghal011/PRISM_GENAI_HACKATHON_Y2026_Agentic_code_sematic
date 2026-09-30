# Agentic Code Intelligence

Semantic **code retrieval and ranking**: given a natural-language developer query, return the most relevant code
snippets (file, function, line range, commit) ranked by relevance. It is the retrieval layer of an agentic coding
assistant. It does not generate code or answers.

> **Read this first: what has and has not been measured.**
> Everything below was built and tested in a sandbox with **no access to HuggingFace**. That means:
> * **No CoIR AppsRetrieval numbers exist yet.** `scripts/evaluate.py --mode mteb` is implemented but has *not* been run against the real dataset.
> * The dense encoder used for all measurements is a **CPU-only TF-IDF + SVD (LSA) fallback**, not a pretrained model (BGE/E5/...). Switching is one config line.
> * All accuracy/ablation numbers come from a **local proxy benchmark** (docstring -> function over installed Python source). They are real measurements, but on a different task from AppsRetrieval, and must not be compared with CoIR/MTEB leaderboards.

## Architecture

```
Query -> Preprocess (identifiers, files, libs, concept expansion) -> Classify (12 types)
      -> Hybrid candidates: Dense ANN (FAISS HNSW) + BM25 + exact-identifier index
      -> Normalise + fuse (weights re-scaled per query type) -> top-N
      -> Code-aware rerank (linear | LambdaMART | cross-encoder) -> Version/duplicate filter -> Top-K
```

Diagram: [docs/architecture.md](docs/architecture.md). Layout: `src/aci/{ingestion,query,retrieval,indexing,ranking,evaluation,api}`, CLI in `scripts/`, UI in `demo/`.

Key decisions (and why):
* **Chunk by AST** (function / method / class summary / module block; long functions split into blocks that keep parent metadata). Python uses `ast`; JS/TS/Java/Go/C/C++/C#/Rust/Kotlin/PHP/Scala use a brace-matching heuristic parser (no native deps, CPU-cheap, less exact than tree-sitter).
* **Three text representations** (raw / metadata / contextual). Contextual (path + names + signature + doc + imports + calls + code) won on the proxy benchmark and is the default.
* **Cache keys are content-addressed** (`repo|file|qualname|code-hash` + model version), *without* the commit hash, so an unchanged function is never re-embedded on a new commit.
* **BM25 is a sparse-matrix implementation** (identifiers kept whole *and* split; light symmetric stemmer), fast enough for 100K+ chunks; a separate exact-identifier index rewards names a chunk *defines* (1.0), file stems (0.8) and names it *calls* (0.5).
* **CPU-first**: FAISS HNSW above `flat_threshold` chunks, exact flat search below it; NumPy fallback if FAISS is missing.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
# optional: real embeddings + official MTEB run
pip install sentence-transformers mteb datasets
python -m pytest -q          # 47 tests
```

## Index

```bash
python index.py --repo ./myrepo --commit HEAD                       # scripts/build_index.py
python index.py --repo ./myrepo --commit abc123 --label commit_002  # incremental from the latest indexed version
python index.py --repo ./myrepo --commit abc123 --base commit_001   # explicit base
```
Incremental flow: `git diff base..commit` -> reparse only changed files -> re-embed only new/changed chunks (others come from the cache) -> rebuild BM25/ANN structures (cheap). Plain directories are indexed as a single `worktree` version.

## Search

```bash
python search.py --repo myrepo --query "Where is the input normalized before main?" --top-k 10
python search.py --repo myrepo --query "retry logic" --version commit_002
python search.py --repo myrepo --query "normalize input" --version all        # cross-version, duplicates collapsed
python search.py --repo myrepo --query "normalize input" --version all --history
```
Python API: `aci.api.app.retrieve(query, repo, version="commit_002", top_k=10)`.
UI: `streamlit run demo/streamlit_app.py` (repository, version, query, top-k; shows file, function, lines, score, retrieval channel, commit, highlighted code, ranking features).

## Evaluation

```bash
# Official (needs HuggingFace access; NOT yet run, see the box at the top):
python evaluate.py --mode mteb --task AppsRetrieval --out results/mteb
python evaluate.py --mode mteb --set embedding_model.backend=sentence_transformers
# Offline proxy benchmark:
tar xzf data/local_bench.tar.gz -C data     # the exact benchmark used for the numbers below
# (scripts/make_local_benchmark.py regenerates one from *your* installed Python packages, so its corpus and numbers will differ)
python evaluate.py --mode local
python scripts/ablation.py       # ablations, weight tuning, LambdaMART, version experiment
python benchmark.py              # latency at 1K/10K/50K/100K snippets
```
`ACISearchModel` (src/aci/evaluation/mteb_model.py) implements mteb 2.x's `SearchProtocol`, so MTEB scores the **whole hybrid pipeline**, not only an embedder. Tested here: the `index()/search()` contract with in-memory datasets and `ModelMeta` construction. **Not tested:** a full `mteb.evaluate` run, which needs the dataset download.

### Dataset and metrics
**CoIR AppsRetrieval** (MTEB task `AppsRetrieval`, from the APPS benchmark): natural-language programming problem statements as queries, Python solutions as the corpus; main score NDCG@10. Corpus items are whole scripts, so each is one retrieval unit and function-name/structural features will matter less there than on real repositories.
* **MRR@10**: mean of 1/rank of the first relevant result (0 if none in the top 10).
* **NDCG@10**: discounted cumulative gain of the top 10 (gain / log2(rank+1)) divided by the ideal ordering's.
Metrics are implemented in `evaluation/metrics.py` (cross-checked in a unit test against hand-computed values; against `pytrec_eval` if installed).

## Measured results (local proxy benchmark, not AppsRetrieval)

Setup: 20,000 real functions from installed Python packages, docstrings **removed** from the indexed code; the first sentence of each docstring is the query; one relevant function per query. 500 queries: 250 for tuning (val), **250 held out for the numbers below**. LSA (128-d) encoder, 1 CPU core. Raw output: `results/ablation_local.json`.

| Approach | MRR@10 | NDCG@10 | Recall@100 |
|---|---|---|---|
| 1. BM25 only | 0.4701 | 0.5183 | 0.8160 |
| 2. Dense only (LSA) | 0.1399 | 0.1653 | 0.5080 |
| 3. Dense + BM25 (default weights) | 0.3637 | 0.4082 | 0.7880 |
| 4. + identifier channel | 0.3550 | 0.3990 | 0.7880 |
| 5. + reranker (basic features) | 0.3301 | 0.3724 | 0.7880 |
| 6. + query preprocessing | 0.3221 | 0.3625 | 0.7840 |
| 7. + metadata features (different reranker weights) | 0.3863 | 0.4295 | 0.7840 |
| 8. + structural features (Full, default weights) | 0.3501 | 0.3917 | 0.7840 |
| 9. Full system, weights tuned on val | 0.4884 | 0.5325 | 0.8160 |
| 10. Full features + LambdaMART (trained on val) | 0.4962 | 0.5315 | 0.7840 |

Representation experiment (which text is embedded/indexed):

| Representation | Dense only MRR / NDCG | Hybrid MRR / NDCG |
|---|---|---|
| raw code | 0.1246 / 0.1557 | 0.3079 / 0.3455 |
| metadata | 0.0469 / 0.0606 | 0.2737 / 0.3083 |
| contextual | 0.1399 / 0.1653 | 0.3637 / 0.4082 |

What the numbers do and do not say:
* With the LSA fallback, **default weights make hybrid *worse* than BM25 alone** (rows 1 vs 3-8): the dense channel is weak and the default 0.55 dense weight lets it drag results down. That is why the default config is a starting point, not a result.
* Tuned weights (row 9, best random-search trial on val, applied to the held-out half) put ~0.67 on BM25 and reach 0.5325 NDCG@10 vs 0.5183 for BM25 alone (+0.014). **With 250 queries I did not test significance**, so treat this as "at least matches BM25", not a proven gain. LambdaMART (row 10) is likewise within noise of row 9. The tuned profile is saved in `config/config.lsa_tuned.yaml`.
* **Query concept expansion hurt here** (row 5 -> 6). These queries are docstrings that already share vocabulary with the code; expansion may help queries that don't, but that is untested. **Structural features also lowered the score** (row 7 -> 8): docstring queries rarely name a function whose neighbours are relevant, so this benchmark cannot show their value. Rows 7 vs 6 change the reranker weights as well as adding metadata features, so that +0.067 NDCG is not a clean single-feature ablation.
* Per the spec ("remove components that make performance worse"): on this proxy, expansion and structural features are candidates for removal; both stay switchable in config (`query.expand_concepts`, `ranking.use_structural`) because they target query types this benchmark does not contain.

Version-aware retrieval (simulated, `results/ablation_local.json`): 4,000 units, a second version in which 30% were refactored (rename + comment); 127 queries searched across both versions.

| | unit MRR@10 | duplicate rate in top 10 |
|---|---|---|
| no dedup | 0.3317 | 0.4795 |
| version-aware dedup | 0.3725 | 0.0 |

Incremental indexing on that experiment: the second version embedded 1,200 of 4,000 chunks (the 30% that changed); 2,800 came from cache. On the 3-commit test repo, indexing commit 3 embedded exactly 1 chunk (unit test).
The refactors are synthetic (a rename plus a trailing comment); real history is messier.

### Latency (1 CPU core, LSA encoder, proxy corpus of real Python functions; 200 queries, times in ms)

| Snippets | build (s) | ANN | BM25 | query embed | preprocess | rerank | total mean | total p95 |
|---|---|---|---|---|---|---|---|---|
| 1,000 | 3.2 | 0.16 | 0.28 | 1.24 | 0.25 | 3.42 | 6.07 | 7.59 |
| 10,000 | 10.0 | 0.17 | 0.17 | 1.08 | 0.14 | 1.77 | 3.78 | 4.34 |
| 50,000 | 34.6 | 0.20 | 0.20 | 2.13 | 0.14 | 1.78 | 6.25 | 5.89 |
| 100,000 | 58.5 | 0.23 | 0.24 | 2.32 | 0.14 | 1.86 | 5.47 | 6.30 |

Retrieval cost is nearly flat in corpus size (ANN and BM25 stay well under 1 ms). The 1K row is slower than 10K, which I did not investigate (likely warm-up/noise). The stage columns do not sum to the total (candidate scoring and feature assembly are not timed separately). **A pretrained encoder will add model inference time to "query embed" that is not measured here.**

## Demo queries
`python scripts/demo_queries.py` runs the seven spec queries on the hand-written `examples/sample_repo` (5 files), output in `results/demo_queries.md`. This is a smoke test of behaviour, **not** evidence of accuracy: the sample was written to contain the answers. All seven top-1 results were the intended function.

Reproducibility: results were produced on 1 CPU core with the shipped `data/local_bench.tar.gz`, `config/config.yaml` and seeds fixed at 0; weight tuning uses random search, so re-runs on different hardware/library versions may differ slightly.

## Error handling
Empty query, missing repository, invalid commit, unsupported language (skipped and counted), corrupt/unparseable files (fall back to line windows, or skipped), very large files (`chunking.max_file_bytes`), missing/incomplete index, embedder/index mismatch, model-loading failure (falls back to LSA when `fallback_to_lsa: true`), duplicate chunks. Covered by `tests/`.

## Limitations
* No real AppsRetrieval score yet; see the box at the top. Expect the dense encoder choice to matter far more there than anything measured here.
* The LSA encoder is a weak substitute for a pretrained model. `sentence_transformers` and `cross_encoder` code paths are implemented but **untested here** (no model download).
* Non-Python parsing is a brace-matching heuristic, exercised only by a JavaScript unit test. Call-graph edges are by callee *name*, unresolved across modules/aliases.
* BM25 and ANN structures are rebuilt for each version (only parsing and embedding are incremental); HNSW has no deletions.
* Cross-version dedup uses identity, embedding similarity and body-token overlap heuristics; renames plus large edits will not be recognised.
* The category weights and concept map are hand-written, not learned. Tuned weights come from one random search on 250 queries of one synthetic task.
