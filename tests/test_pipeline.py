from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from rag_qa.chunking import chunk_documents
from rag_qa.embeddings import HashingEmbeddings
from rag_qa.loaders import load_documents
from rag_qa.pipeline import RAGPipeline, is_abstention
from rag_qa.prompts import ANSWER_PROMPT, NO_ANSWER
from rag_qa.rerank import LLMReranker, NoopReranker
from rag_qa.vectorstore import load_index

from .conftest import DOCS_DIR, StructuredStub


def test_loader_reads_all_markdown():
    docs = load_documents(DOCS_DIR)
    assert {d.metadata["source"] for d in docs} >= {"employee_handbook.md", "cloud_api_reference.md"}


def test_loader_rejects_empty_dir(tmp_path: Path):
    with pytest.raises(ValueError):
        load_documents(tmp_path)


def test_chunks_respect_size_and_carry_metadata():
    docs = load_documents(DOCS_DIR)
    chunks = chunk_documents(docs, chunk_size=400, chunk_overlap=50)
    assert len(chunks) > len(docs)
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))
    for c in chunks:
        body = c.page_content.split("\n", 1)[-1]
        assert len(body) <= 400
        assert c.metadata["source"]
    # Section headers are attached as context
    assert any("Paid Time Off" in c.metadata["section"] for c in chunks)


def test_chunk_overlap_validation():
    with pytest.raises(ValueError):
        chunk_documents([Document(page_content="x", metadata={"source": "a"})], 100, 100)


def test_hashing_embeddings_normalised_and_deterministic():
    emb = HashingEmbeddings(dim=256)
    a, b = emb.embed_query("rate limit pro plan"), emb.embed_query("rate limit pro plan")
    assert a == b
    assert abs(sum(x * x for x in a) - 1.0) < 1e-9


def test_index_roundtrip_and_manifest_guard(settings, built_index):
    store = load_index(settings, embeddings=HashingEmbeddings())
    hits = store.similarity_search("Pro plan requests per minute", k=3)
    assert any(h.metadata["source"] == "cloud_api_reference.md" for h in hits)
    with pytest.raises(ValueError, match="built with"):
        load_index(settings.with_overrides(embedding_provider="huggingface"), embeddings=HashingEmbeddings())


def test_vector_search_scores_are_cosine_and_sorted(settings, built_index):
    store, _ = built_index
    pipe = RAGPipeline(settings, vectorstore=store, llm=FakeListChatModel(responses=["x"]), reranker=NoopReranker())
    hits = pipe.vector_search("How many PTO days per year?", k=5)
    scores = [h.vector_score for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert all(-1.0 <= s <= 1.0 for s in scores)
    assert hits[0].source == "employee_handbook.md"


def test_answer_extracts_citations_and_contexts(settings, built_index):
    store, _ = built_index
    llm = FakeListChatModel(responses=["Full-time employees get 21 days of PTO per year [1][3]."])
    pipe = RAGPipeline(settings, vectorstore=store, llm=llm, reranker=NoopReranker())
    ans = pipe.answer("How many PTO days does a full-time employee get?")
    assert ans.cited == [1, 3]
    assert len(ans.contexts) == settings.top_k
    assert [c.chunk_id for c in ans.cited_chunks] == [ans.contexts[0].chunk_id, ans.contexts[2].chunk_id]
    assert not ans.abstained


def test_abstention_detection():
    assert is_abstention(NO_ANSWER)
    assert is_abstention("I don’t know based on the provided documents.")
    assert not is_abstention("The limit is 600 requests per minute [1].")


def test_answer_prompt_contains_grounding_rules():
    msgs = ANSWER_PROMPT.format_messages(context="[1] ctx", question="q?")
    assert "ONLY" in msgs[0].content and NO_ANSWER in msgs[0].content
    assert msgs[1].content == "Question: q?"


def test_llm_reranker_orders_by_score_with_stable_ties():
    docs = [Document(page_content=f"passage {i}") for i in range(4)]

    def fake(schema, _prompt):
        return schema(scores=[{"passage": 1, "score": 1}, {"passage": 2, "score": 3},
                              {"passage": 3, "score": 1}, {"passage": 4, "score": 0}])

    ranked = LLMReranker(StructuredStub(fake)).rerank("q", docs, top_k=3)
    assert [d.page_content for d, _ in ranked] == ["passage 1", "passage 0", "passage 2"]
