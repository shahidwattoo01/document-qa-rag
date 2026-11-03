# Retrieval sweep (cross-encoder re-ranking vs. vector-only)

43 answerable questions, fetch_k=20, re-ranker `cross-encoder/ms-marco-MiniLM-L-6-v2`.

| Embeddings | k | Vector Recall@k | Re-ranked Recall@k | Vector MRR | Re-ranked MRR | Vector nDCG@k | Re-ranked nDCG@k |
|---|---|---|---|---|---|---|---|
| hashing-1024 | 1 | 0.663 | 0.884 | 0.674 | 0.907 | 0.674 | 0.907 |
| hashing-1024 | 2 | 0.814 | 0.965 | 0.744 | 0.942 | 0.762 | 0.942 |
| hashing-1024 | 4 | 0.895 | 0.965 | 0.769 | 0.942 | 0.800 | 0.942 |
| sentence-transformers/all-MiniLM-L6-v2 | 1 | 0.884 | 0.907 | 0.907 | 0.930 | 0.907 | 0.930 |
| sentence-transformers/all-MiniLM-L6-v2 | 2 | 0.965 | 0.988 | 0.942 | 0.965 | 0.942 | 0.965 |
| sentence-transformers/all-MiniLM-L6-v2 | 4 | 0.988 | 0.988 | 0.948 | 0.965 | 0.952 | 0.965 |
