# Agentic Code Intelligence: slide content (10 slides)

Numbers are from the local proxy benchmark (docstring -> function, 20K Python functions, 250 held-out queries, LSA fallback encoder). **No AppsRetrieval result exists yet; do not present these as CoIR/MTEB scores.** Fill slide 9's AppsRetrieval row after running `python evaluate.py --mode mteb`.

**1. Title**: Agentic Code Intelligence. Semantic code retrieval and ranking, CPU-first, version-aware.

**2. Problem**: Codebases have 10^4 to 10^6 snippets; names are unreliable (`perf`, `check`); grep misses intent, embeddings alone miss exact identifiers; code changes every commit, so indexes go stale. Example: "How is the input preprocessed before going to main?" -> `normalize` > `check` > `perf`.

**3. Objective**: Natural-language query -> ranked relevant code (file, function, lines, commit). Priorities: retrieval accuracy (NDCG@10, MRR) > version support > latency > complexity. Retrieval only; generation is a downstream layer.

**4. Architecture**: Query -> preprocess -> classify -> [dense ANN + BM25 + identifier] -> fuse -> code-aware rerank -> version/duplicate filter -> Top-K. (Use the diagram in docs/architecture.md.)

**5. Query and code processing**: Query: identifiers, files, libs, actions kept whole; 12 query types (rule-based) re-weight channels. Code: AST chunks (function/method/class/block), each with path, names, signature, doc, imports, calls, line range, commit. Three representations; contextual won (hybrid NDCG@10 0.408 vs raw 0.346 vs metadata 0.308).

**6. Retrieval**: Dense (FAISS HNSW), BM25 (identifier-aware), exact-identifier index; min-max normalised weighted fusion (or RRF). Honest finding: with the weak LSA encoder, default hybrid weights (NDCG@10 0.408) *underperformed* BM25 alone (0.518); tuned weights reach 0.533. A pretrained encoder is the expected fix, untested here.

**7. Reranking**: Top-N only. Transparent linear score over features: semantic, BM25, identifier, function/class/file-path match, doc overlap, metadata, call-graph neighbour. Optional LambdaMART (NDCG@10 0.532 on the proxy, within noise of tuned linear) and cross-encoder (untested). Ablation: concept expansion and structural features did not help on this proxy.

**8. Version-aware retrieval**: Index per commit; `git diff` -> reparse changed files -> re-embed changed chunks only (content-addressed cache: 2,800 of 4,000 embeddings reused when 30% changed). Cross-version search collapses renamed/duplicate units: duplicate rate in top-10 0.48 -> 0.0, unit MRR@10 0.332 -> 0.373 (simulated refactors).

**9. Evaluation**: Table of the 10 ablation rows (see README). Latency (1 CPU): 1K 6.1 ms, 10K 3.8 ms, 50K 6.3 ms, 100K 5.5 ms total mean; ANN and BM25 < 0.25 ms even at 100K. AppsRetrieval: [pending run]. State clearly that the proxy is not AppsRetrieval.

**10. Demo and future work**: Streamlit demo; 7 difficult queries on the sample repo (results/demo_queries.md; caveat: hand-written sample). Future: pretrained code embedder (BGE/E5/code models) and re-tuned weights; tree-sitter parsing; resolved call graph; learned query classifier; run the official AppsRetrieval evaluation; hand the results to an LLM agent.
