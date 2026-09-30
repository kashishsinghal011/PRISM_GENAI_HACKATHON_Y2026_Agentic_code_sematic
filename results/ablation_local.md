| Approach | MRR@10 | NDCG@10 | Recall@100 | mean latency (ms) |
|---|---|---|---|---|
| 1. BM25 only | 0.4701 | 0.5183 | 0.8160 | 4.84 |
| 2. Dense only (LSA) | 0.1399 | 0.1653 | 0.5080 | 4.18 |
| 3. Dense + BM25 (hybrid) | 0.3637 | 0.4082 | 0.7880 | 4.54 |
| 4. Hybrid + identifier channel | 0.3550 | 0.3990 | 0.7880 | 4.74 |
| 5. Hybrid + reranker (basic features) | 0.3301 | 0.3724 | 0.7880 | 4.74 |
| 6. + query preprocessing | 0.3221 | 0.3625 | 0.7840 | 4.69 |
| 7. + metadata features | 0.3863 | 0.4295 | 0.7840 | 4.7 |
| 8. + structural features (Full, default weights) | 0.3501 | 0.3917 | 0.7840 | 4.73 |
| 9. Full system, weights tuned on val | 0.4884 | 0.5325 | 0.8160 | 4.72 |
| 10. Full features + LambdaMART (trained on val) | 0.4962 | 0.5315 | 0.7840 | 6.3 |

| Representation | System | MRR@10 | NDCG@10 |
|---|---|---|---|
| raw | dense only | 0.1246 | 0.1557 |
| raw | hybrid | 0.3079 | 0.3455 |
| metadata | dense only | 0.0469 | 0.0606 |
| metadata | hybrid | 0.2737 | 0.3083 |
| contextual | dense only | 0.1399 | 0.1653 |
| contextual | hybrid | 0.3637 | 0.4082 |
