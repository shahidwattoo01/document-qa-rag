"""Retrieval-only sweep over embeddings x re-ranker x k, runnable without an OpenAI key.

    python scripts/retrieval_sweep.py            # writes results/retrieval_sweep.md

Builds a fresh in-memory index per embedding backend and compares vector-only ranking
with cross-encoder re-ranking at several cutoffs on the labeled question set.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_qa.config import PROJECT_ROOT, get_settings  # noqa: E402
from rag_qa.evaluate import evaluate_retrieval, load_qa_set  # noqa: E402
from rag_qa.pipeline import RAGPipeline  # noqa: E402
from rag_qa.rerank import CrossEncoderReranker  # noqa: E402
from rag_qa.vectorstore import build_index  # noqa: E402

EMBEDDINGS = sys.argv[1:] or ["hashing", "huggingface"]
KS = [1, 2, 4]


def main() -> None:
    items = load_qa_set()
    reranker = CrossEncoderReranker()
    rows = []
    for provider in EMBEDDINGS:
        settings = get_settings(embedding_provider=provider, embedding_model="")
        store, _ = build_index(settings, save=False)
        pipe = RAGPipeline(settings, vectorstore=store, reranker=reranker)
        for k in KS:
            r = evaluate_retrieval(pipe, items, top_k=k)
            rows.append((settings.resolved_embedding_model, k, r))
            print(f"{provider} k={k}: vector recall {r['vector']['recall']:.3f} mrr {r['vector']['mrr']:.3f} | "
                  f"rerank recall {r['reranked']['recall']:.3f} mrr {r['reranked']['mrr']:.3f}", flush=True)

    lines = [
        "# Retrieval sweep (cross-encoder re-ranking vs. vector-only)",
        "",
        f"{rows[0][2]['n']} answerable questions, fetch_k={rows[0][2]['fetch_k']}, "
        "re-ranker `cross-encoder/ms-marco-MiniLM-L-6-v2`.",
        "",
        "| Embeddings | k | Vector Recall@k | Re-ranked Recall@k | Vector MRR | Re-ranked MRR | Vector nDCG@k | Re-ranked nDCG@k |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for model, k, r in rows:
        v, rr = r["vector"], r["reranked"]
        lines.append(f"| {model} | {k} | {v['recall']:.3f} | {rr['recall']:.3f} | {v['mrr']:.3f} | "
                     f"{rr['mrr']:.3f} | {v['ndcg']:.3f} | {rr['ndcg']:.3f} |")
    out = PROJECT_ROOT / "results" / "retrieval_sweep.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
