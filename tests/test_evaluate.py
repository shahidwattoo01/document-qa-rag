import json

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from rag_qa.evaluate import Judge, QAItem, evaluate_generation, render_markdown, retrieval_metrics, run_evaluation
from rag_qa.pipeline import RAGPipeline
from rag_qa.prompts import NO_ANSWER
from rag_qa.rerank import NoopReranker

from .conftest import StructuredStub

EV = ["alpha beta", "gamma delta"]


def test_retrieval_metrics_perfect():
    m = retrieval_metrics(["xx alpha beta", "gamma delta yy", "zz"], EV, k=3)
    assert m == {"hit": 1.0, "recall": 1.0, "mrr": 1.0, "ndcg": pytest.approx(1.0), "first_relevant_rank": 1}


def test_retrieval_metrics_partial_and_late():
    m = retrieval_metrics(["nothing", "nope", "Alpha   BETA here"], EV, k=3)
    assert m["hit"] == 1.0 and m["recall"] == 0.5
    assert m["mrr"] == pytest.approx(1 / 3)
    assert 0 < m["ndcg"] < 1


def test_retrieval_metrics_respects_k_and_misses():
    m = retrieval_metrics(["nothing", "alpha beta"], EV, k=1)
    assert m["hit"] == 0.0 and m["mrr"] == 0.0 and m["recall"] == 0.0


def test_duplicate_evidence_not_double_counted():
    m = retrieval_metrics(["alpha beta", "alpha beta again"], EV, k=2)
    assert m["recall"] == 0.5


def _judge():
    def fake(schema, prompt_value):
        text = prompt_value.to_string()
        if schema.__name__ == "_FaithfulnessVerdict":
            return schema(claims=[{"claim": "c1", "supported": True}, {"claim": "c2", "supported": "Mars" not in text}])
        return schema(reasoning="ok", score=0.9)

    return Judge(StructuredStub(fake))


def test_generation_eval_metrics(settings, built_index):
    store, _ = built_index
    items = [
        QAItem(id="a", question="How many PTO days per year?", reference_answer="21 days.",
               answerable=True, evidence=["for a total of 21 days per year"], sources=["employee_handbook.md"]),
        QAItem(id="b", question="Who is the CEO?", reference_answer="Not stated.", answerable=False),
    ]
    llm = FakeListChatModel(responses=["21 days per year [1], also on Mars.", NO_ANSWER])
    pipe = RAGPipeline(settings, vectorstore=store, llm=llm, reranker=NoopReranker())
    res = evaluate_generation(pipe, items, _judge())
    assert res["faithfulness"] == 0.5          # one of two claims unsupported
    assert res["correctness"] == 1.0           # 0.9 snaps to 1.0
    assert res["unanswerable_abstention_rate"] == 1.0
    assert res["false_refusal_rate"] == 0.0
    assert res["per_question"][0]["unsupported_claims"] == ["c2"]


def test_run_evaluation_retrieval_only_writes_reports(settings, built_index, tmp_path):
    store, _ = built_index
    pipe = RAGPipeline(settings, vectorstore=store, llm=FakeListChatModel(responses=["x"]), reranker=NoopReranker())
    report = run_evaluation(pipe, with_generation=False, out_dir=tmp_path)
    run = report["retrieval"][0]
    assert run["n"] == report["num_answerable"]
    # With no re-ranker the two rankings must be identical.
    assert run["vector"] == run["reranked"]
    # Even lexical hashing embeddings should find most evidence in the candidate pool.
    assert run["pool_recall"] > 0.6
    assert json.loads((tmp_path / "eval_report.json").read_text(encoding="utf-8"))["config"]["embedding_provider"] == "hashing"
    md = (tmp_path / "eval_report.md").read_text(encoding="utf-8")
    assert "Recall@k" in md and md == render_markdown(report)
