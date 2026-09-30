# Architecture

```mermaid
flowchart TD
    Q[User query] --> P[Query preprocessor<br/>identifiers, files, libs, concept expansion]
    P --> C[Query classifier<br/>12 categories, rule-based]
    C -->|per-category channel weights| H
    subgraph H[Candidate generation, top-N configurable]
      D[Dense ANN<br/>FAISS HNSW / flat] --- B[BM25<br/>sparse, code-aware tokens] --- I[Exact identifier index<br/>defined / called / file stem]
    end
    H --> F[Min-max normalise + weighted fusion or RRF]
    F --> R[Code-aware reranker on top-N<br/>linear | LambdaMART | cross-encoder]
    R --> V[Version / duplicate filter]
    V --> K[Top-K + explanation]
    subgraph IDX[Indexing, per version]
      G[git diff / blob compare] --> PA[Parse changed files only<br/>ast + brace parser] --> CH[Chunks + metadata] --> EM[Embed only new chunks<br/>content-addressed cache] --> S[(indexes/repo/version/)]
    end
    S -.-> H
```

Per-version index layout: `indexes/<repo>/<label>/{manifest.json, chunks.jsonl.gz, vectors.npy, faiss.index, bm25.pkl}` plus
`indexes/<repo>/refs.json` (label -> commit) and one shared `embedder_lsa.pkl` (frozen so vectors stay comparable across commits).
