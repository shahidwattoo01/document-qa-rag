"""Second-stage re-rankers.

FAISS retrieval is a bi-encoder: query and passage are embedded independently, which is
fast but coarse. Re-rankers look at the (query, passage) pair jointly and re-order a
larger candidate pool (``fetch_k``) down to the ``top_k`` passages sent to the LLM.

- ``CrossEncoderReranker``: ms-marco MiniLM cross-encoder, runs locally on CPU.
- ``LLMReranker``: asks the chat model to grade each candidate 0-3 (structured output).
- ``NoopReranker``: keeps the vector-similarity order (baseline).
"""

from __future__ import annotations

from typing import Protocol

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from .config import Settings
from .prompts import RERANK_PROMPT


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, docs: list[Document], top_k: int) -> list[tuple[Document, float]]: ...


class NoopReranker:
    name = "none"

    def rerank(self, query: str, docs: list[Document], top_k: int) -> list[tuple[Document, float]]:
        n = len(docs)
        return [(d, float(n - i)) for i, d in enumerate(docs[:top_k])]


class CrossEncoderReranker:
    name = "cross-encoder"

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, docs: list[Document], top_k: int) -> list[tuple[Document, float]]:
        if not docs:
            return []
        scores = self.model.predict([(query, d.page_content) for d in docs])
        ranked = sorted(zip(docs, (float(s) for s in scores)), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


class _PassageScore(BaseModel):
    passage: int = Field(description="Passage number as shown in the prompt")
    score: int = Field(ge=0, le=3, description="Relevance score 0-3")


class _RerankResult(BaseModel):
    scores: list[_PassageScore]


class LLMReranker:
    name = "llm"

    def __init__(self, llm: BaseChatModel):
        self.chain = RERANK_PROMPT | llm.with_structured_output(_RerankResult)

    def rerank(self, query: str, docs: list[Document], top_k: int) -> list[tuple[Document, float]]:
        if not docs:
            return []
        passages = "\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(docs, 1))
        result: _RerankResult = self.chain.invoke({"question": query, "passages": passages})
        by_idx = {s.passage: s.score for s in result.scores}
        # Ties (common with a 0-3 scale) are broken by the original vector rank.
        scored = [
            (d, by_idx.get(i, 0) + (len(docs) - i) / (10 * len(docs))) for i, d in enumerate(docs, 1)
        ]
        return sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]


def get_reranker(settings: Settings, llm: BaseChatModel | None = None, name: str | None = None) -> Reranker:
    name = name or settings.reranker
    if name == "none":
        return NoopReranker()
    if name == "cross-encoder":
        return CrossEncoderReranker(settings.cross_encoder_model)
    if name == "llm":
        if llm is None:
            from .pipeline import make_llm

            llm = make_llm(settings.llm_model, settings.temperature)
        return LLMReranker(llm)
    raise ValueError(f"Unknown reranker: {name!r} (expected cross-encoder | llm | none)")
