"""Evaluation against the hand-labeled question set.

Retrieval relevance is defined by *evidence spans*, not chunk IDs: each labeled question
lists one or more verbatim sentences from the source documents that are needed to answer
it, and a retrieved chunk is relevant if it contains one of them. This keeps the labels
valid when chunk size / overlap change, so different chunking configs can be compared.

Retrieval metrics (at k = top_k):
  hit@k     - at least one evidence span retrieved
  recall@k  - fraction of the question's evidence spans retrieved (matters for multi-hop)
  MRR       - 1 / rank of the first relevant chunk
  nDCG@k    - rank-aware gain, one unit per distinct evidence span

Generation metrics (LLM-as-judge, needs an LLM):
  faithfulness  - share of the answer's claims supported by the retrieved context
  correctness   - 0 / 0.5 / 1 agreement with the human reference answer
  abstention    - unanswerable questions: did the model refuse? answerable: did it NOT refuse?
  citation_hit  - answer cites at least one chunk that contains evidence
"""

from __future__ import annotations

import json
import math
import re
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from .config import PROJECT_ROOT
from .pipeline import RAGAnswer, RAGPipeline, RetrievedChunk, format_context
from .prompts import CORRECTNESS_PROMPT, FAITHFULNESS_PROMPT

DEFAULT_QA_PATH = PROJECT_ROOT / "data" / "eval" / "qa_set.jsonl"
RESULTS_DIR = PROJECT_ROOT / "results"

_WS = re.compile(r"\s+")


# ---------------------------------------------------------------------------- data
@dataclass
class QAItem:
    id: str
    question: str
    reference_answer: str
    answerable: bool
    category: str = "factual"
    sources: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)


def load_qa_set(path: Path | str = DEFAULT_QA_PATH) -> list[QAItem]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(QAItem(**json.loads(line)))
    return items


def normalize(text: str) -> str:
    return _WS.sub(" ", text.replace("’", "'")).strip().lower()


def evidence_hits(chunk_text: str, evidence: list[str]) -> set[int]:
    """Indices of evidence spans contained in the chunk."""
    norm = normalize(chunk_text)
    return {i for i, ev in enumerate(evidence) if normalize(ev) in norm}


# ----------------------------------------------------------------- retrieval metrics
def retrieval_metrics(chunks: list[RetrievedChunk] | list[str], evidence: list[str], k: int) -> dict:
    texts = [c if isinstance(c, str) else c.text for c in chunks][:k]
    found: set[int] = set()
    first_rank = None
    dcg = 0.0
    for rank, text in enumerate(texts, 1):
        hits = evidence_hits(text, evidence)
        new = hits - found
        if hits and first_rank is None:
            first_rank = rank
        dcg += len(new) / math.log2(rank + 1)
        found |= hits
    n = len(evidence)
    idcg = sum(1 / math.log2(r + 1) for r in range(1, min(n, k) + 1))
    return {
        "hit": float(bool(found)),
        "recall": len(found) / n if n else 0.0,
        "mrr": 1.0 / first_rank if first_rank else 0.0,
        "ndcg": dcg / idcg if idcg else 0.0,
        "first_relevant_rank": first_rank,
    }


def _mean(values: list[float]) -> float:
    return round(statistics.fmean(values), 4) if values else float("nan")


def evaluate_retrieval(
    pipeline: RAGPipeline,
    items: list[QAItem],
    top_k: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Compare the vector-only ranking with the re-ranked ranking on the same candidates."""
    k = top_k or pipeline.settings.top_k
    rows = []
    answerable = [q for q in items if q.answerable]
    for i, q in enumerate(answerable, 1):
        candidates = pipeline.vector_search(q.question, pipeline.settings.fetch_k)
        reranked = pipeline.reranker.rerank(q.question, [c.document for c in candidates], k)
        rows.append(
            {
                "id": q.id,
                "category": q.category,
                "vector": retrieval_metrics(candidates, q.evidence, k),
                "reranked": retrieval_metrics([d.page_content for d, _ in reranked], q.evidence, k),
                # Upper bound for any re-ranker: is the evidence in the candidate pool at all?
                "pool_recall": retrieval_metrics(candidates, q.evidence, len(candidates))["recall"],
            }
        )
        if progress:
            progress(f"retrieval {i}/{len(answerable)}")

    def agg(key: str, subset: list[dict]) -> dict:
        return {
            m: _mean([r[key][m] for r in subset]) for m in ("hit", "recall", "mrr", "ndcg")
        }

    by_cat = {}
    for cat in sorted({r["category"] for r in rows}):
        subset = [r for r in rows if r["category"] == cat]
        by_cat[cat] = {"n": len(subset), "vector": agg("vector", subset), "reranked": agg("reranked", subset)}

    return {
        "k": k,
        "fetch_k": pipeline.settings.fetch_k,
        "reranker": pipeline.reranker.name,
        "n": len(rows),
        "vector": agg("vector", rows),
        "reranked": agg("reranked", rows),
        "pool_recall": _mean([r["pool_recall"] for r in rows]),
        "by_category": by_cat,
        "per_question": rows,
    }


# ---------------------------------------------------------------- generation metrics
class _Claim(BaseModel):
    claim: str
    supported: bool


class _FaithfulnessVerdict(BaseModel):
    claims: list[_Claim] = Field(default_factory=list)


class _CorrectnessVerdict(BaseModel):
    reasoning: str = Field(description="One or two sentences")
    score: float = Field(description="0.0, 0.5, or 1.0")


class Judge:
    def __init__(self, llm: BaseChatModel):
        self.faith_chain = FAITHFULNESS_PROMPT | llm.with_structured_output(_FaithfulnessVerdict)
        self.correct_chain = CORRECTNESS_PROMPT | llm.with_structured_output(_CorrectnessVerdict)

    def faithfulness(self, answer: RAGAnswer) -> dict:
        if answer.abstained:
            return {"score": None, "claims": []}
        verdict: _FaithfulnessVerdict = self.faith_chain.invoke(
            {"context": format_context(answer.contexts), "answer": answer.answer}
        )
        claims = verdict.claims
        score = sum(c.supported for c in claims) / len(claims) if claims else None
        return {"score": score, "claims": [c.model_dump() for c in claims]}

    def correctness(self, item: QAItem, answer: RAGAnswer) -> dict:
        verdict: _CorrectnessVerdict = self.correct_chain.invoke(
            {"question": item.question, "reference": item.reference_answer, "answer": answer.answer}
        )
        score = min((0.0, 0.5, 1.0), key=lambda v: abs(v - verdict.score))
        return {"score": score, "reasoning": verdict.reasoning}


def evaluate_generation(
    pipeline: RAGPipeline,
    items: list[QAItem],
    judge: Judge,
    progress: Callable[[str], None] | None = None,
) -> dict:
    rows = []
    for i, q in enumerate(items, 1):
        ans = pipeline.answer(q.question)
        row = {
            "id": q.id,
            "category": q.category,
            "answerable": q.answerable,
            "question": q.question,
            "answer": ans.answer,
            "abstained": ans.abstained,
            "abstention_correct": ans.abstained != q.answerable,
            "contexts": [c.chunk_id for c in ans.contexts],
            "cited": [c.chunk_id for c in ans.cited_chunks],
            "latency_s": round(ans.latency_s, 3),
        }
        faith = judge.faithfulness(ans)
        row["faithfulness"] = faith["score"]
        row["unsupported_claims"] = [c["claim"] for c in faith["claims"] if not c["supported"]]
        if q.answerable:
            row["correctness"] = judge.correctness(q, ans)["score"]
            row["citation_hit"] = float(
                any(evidence_hits(c.text, q.evidence) for c in ans.cited_chunks)
            )
        rows.append(row)
        if progress:
            progress(f"generation {i}/{len(items)}")

    ans_rows = [r for r in rows if r["answerable"]]
    na_rows = [r for r in rows if not r["answerable"]]
    faith_scores = [r["faithfulness"] for r in rows if r["faithfulness"] is not None]
    return {
        "n": len(rows),
        "faithfulness": _mean(faith_scores),
        "fully_faithful_rate": _mean([float(s == 1.0) for s in faith_scores]),
        "correctness": _mean([r["correctness"] for r in ans_rows]),
        "citation_hit": _mean([r["citation_hit"] for r in ans_rows]),
        "false_refusal_rate": _mean([float(r["abstained"]) for r in ans_rows]),
        "unanswerable_abstention_rate": _mean([float(r["abstained"]) for r in na_rows]),
        "mean_latency_s": _mean([r["latency_s"] for r in rows]),
        "per_question": rows,
    }


# ------------------------------------------------------------------------ reporting
def _fmt(v) -> str:
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.3f}"


def render_markdown(report: dict) -> str:
    cfg = report["config"]
    lines = [
        "# RAG Evaluation Report",
        "",
        f"- Generated: {report['generated_at']}",
        f"- Questions: {report['num_questions']} ({report['num_answerable']} answerable, "
        f"{report['num_questions'] - report['num_answerable']} unanswerable)",
        f"- Embeddings: `{cfg['embedding_provider']}/{cfg['embedding_model']}`, "
        f"chunk {cfg['chunk_size']}/{cfg['chunk_overlap']}, fetch_k={cfg['fetch_k']}, top_k={cfg['top_k']}",
        "",
    ]
    for run in report["retrieval"]:
        lines += [
            f"## Retrieval — reranker `{run['reranker']}` (k={run['k']})",
            "",
            "| Ranking | Hit@k | Recall@k | MRR | nDCG@k |",
            "|---|---|---|---|---|",
            f"| Vector only (FAISS) | {_fmt(run['vector']['hit'])} | {_fmt(run['vector']['recall'])} | "
            f"{_fmt(run['vector']['mrr'])} | {_fmt(run['vector']['ndcg'])} |",
            f"| + {run['reranker']} re-rank | {_fmt(run['reranked']['hit'])} | {_fmt(run['reranked']['recall'])} | "
            f"{_fmt(run['reranked']['mrr'])} | {_fmt(run['reranked']['ndcg'])} |",
            "",
            f"Candidate-pool recall@{run['fetch_k']} (ceiling for any re-ranker): {_fmt(run['pool_recall'])}",
            "",
            "| Category | n | Vector Recall@k | Re-ranked Recall@k | Vector MRR | Re-ranked MRR |",
            "|---|---|---|---|---|---|",
        ]
        for cat, c in run["by_category"].items():
            lines.append(
                f"| {cat} | {c['n']} | {_fmt(c['vector']['recall'])} | {_fmt(c['reranked']['recall'])} | "
                f"{_fmt(c['vector']['mrr'])} | {_fmt(c['reranked']['mrr'])} |"
            )
        lines.append("")
    gen = report.get("generation")
    if gen:
        lines += [
            f"## Generation — `{cfg['llm_model']}`, judge `{cfg['judge_model']}`, reranker `{cfg['reranker']}`",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Faithfulness (supported claims / claims) | {_fmt(gen['faithfulness'])} |",
            f"| Answers with zero unsupported claims | {_fmt(gen['fully_faithful_rate'])} |",
            f"| Answer correctness vs. reference | {_fmt(gen['correctness'])} |",
            f"| Cites an evidence chunk | {_fmt(gen['citation_hit'])} |",
            f"| Abstained on unanswerable questions | {_fmt(gen['unanswerable_abstention_rate'])} |",
            f"| False refusals on answerable questions | {_fmt(gen['false_refusal_rate'])} |",
            f"| Mean latency (s) | {_fmt(gen['mean_latency_s'])} |",
            "",
        ]
        bad = [r for r in gen["per_question"] if r["unsupported_claims"] or not r["abstention_correct"]]
        if bad:
            lines += ["### Questions to inspect", ""]
            for r in bad:
                why = "; ".join(r["unsupported_claims"]) or (
                    "refused an answerable question" if r["answerable"] else "answered an unanswerable question"
                )
                lines.append(f"- **{r['id']}** {r['question']} — {why}")
            lines.append("")
    return "\n".join(lines)


def run_evaluation(
    pipeline: RAGPipeline,
    qa_path: Path | str = DEFAULT_QA_PATH,
    rerankers: list[str] | None = None,
    with_generation: bool = True,
    judge_llm: BaseChatModel | None = None,
    out_dir: Path | str = RESULTS_DIR,
    progress: Callable[[str], None] | None = None,
) -> dict:
    from .rerank import get_reranker

    items = load_qa_set(qa_path)
    s = pipeline.settings
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "num_questions": len(items),
        "num_answerable": sum(q.answerable for q in items),
        "config": {
            "embedding_provider": s.embedding_provider,
            "embedding_model": s.resolved_embedding_model,
            "chunk_size": s.chunk_size,
            "chunk_overlap": s.chunk_overlap,
            "fetch_k": s.fetch_k,
            "top_k": s.top_k,
            "reranker": pipeline.reranker.name,
            "llm_model": s.llm_model,
            "judge_model": s.judge_model,
        },
        "retrieval": [],
    }

    default_reranker = pipeline.reranker
    for name in rerankers or [default_reranker.name]:
        pipeline._reranker = (
            default_reranker if name == default_reranker.name else get_reranker(s, llm=pipeline._llm, name=name)
        )
        report["retrieval"].append(evaluate_retrieval(pipeline, items, progress=progress))
    pipeline._reranker = default_reranker

    if with_generation:
        from .pipeline import make_llm

        judge = Judge(judge_llm or make_llm(s.judge_model, 0.0))
        report["generation"] = evaluate_generation(pipeline, items, judge, progress=progress)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "eval_report.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def summary_dict(report: dict) -> dict:
    """Compact headline numbers (handy for printing)."""
    out = {}
    for run in report["retrieval"]:
        out[f"vector_recall@{run['k']}"] = run["vector"]["recall"]
        out[f"{run['reranker']}_recall@{run['k']}"] = run["reranked"]["recall"]
        out["vector_mrr"] = run["vector"]["mrr"]
        out[f"{run['reranker']}_mrr"] = run["reranked"]["mrr"]
    if gen := report.get("generation"):
        out.update({k: gen[k] for k in ("faithfulness", "correctness", "unanswerable_abstention_rate")})
    return out


__all__ = [
    "QAItem",
    "Judge",
    "load_qa_set",
    "evidence_hits",
    "retrieval_metrics",
    "evaluate_retrieval",
    "evaluate_generation",
    "run_evaluation",
    "render_markdown",
    "summary_dict",
]
