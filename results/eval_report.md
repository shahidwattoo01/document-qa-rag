# RAG Evaluation Report

- Generated: 2026-10-17T10:46:05
- Questions: 48 (43 answerable, 5 unanswerable)
- Embeddings: `huggingface/sentence-transformers/all-MiniLM-L6-v2`, chunk 800/120, fetch_k=20, top_k=4

## Retrieval — reranker `none` (k=4)

| Ranking | Hit@k | Recall@k | MRR | nDCG@k |
|---|---|---|---|---|
| Vector only (FAISS) | 1.000 | 0.988 | 0.948 | 0.952 |
| + none re-rank | 1.000 | 0.988 | 0.948 | 0.952 |

Candidate-pool recall@20 (ceiling for any re-ranker): 1.000

| Category | n | Vector Recall@k | Re-ranked Recall@k | Vector MRR | Re-ranked MRR |
|---|---|---|---|---|---|
| factual | 25 | 1.000 | 1.000 | 0.930 | 0.930 |
| multi-hop | 2 | 0.750 | 0.750 | 1.000 | 1.000 |
| paraphrase | 16 | 1.000 | 1.000 | 0.969 | 0.969 |

## Retrieval — reranker `cross-encoder` (k=4)

| Ranking | Hit@k | Recall@k | MRR | nDCG@k |
|---|---|---|---|---|
| Vector only (FAISS) | 1.000 | 0.988 | 0.948 | 0.952 |
| + cross-encoder re-rank | 1.000 | 0.988 | 0.965 | 0.965 |

Candidate-pool recall@20 (ceiling for any re-ranker): 1.000

| Category | n | Vector Recall@k | Re-ranked Recall@k | Vector MRR | Re-ranked MRR |
|---|---|---|---|---|---|
| factual | 25 | 1.000 | 1.000 | 0.930 | 1.000 |
| multi-hop | 2 | 0.750 | 0.750 | 1.000 | 1.000 |
| paraphrase | 16 | 1.000 | 1.000 | 0.969 | 0.906 |
