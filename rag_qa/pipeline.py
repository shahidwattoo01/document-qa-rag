"""The end-to-end RAG pipeline: retrieve -> re-rank -> prompt -> generate (with citations)."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from .config import Settings, get_settings
from .prompts import ANSWER_PROMPT, NO_ANSWER
from .rerank import Reranker, get_reranker
from .vectorstore import load_index

_CITATION_RE = re.compile(r"\[(\d+)\]")


def make_llm(model: str, temperature: float = 0.0) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model, temperature=temperature)


@dataclass
class RetrievedChunk:
    document: Document
    vector_rank: int        # 1-based position in the FAISS results
    vector_score: float     # cosine similarity (higher is better)
    rerank_score: float | None = None

    @property
    def source(self) -> str:
        return self.document.metadata.get("source", "?")

    @property
    def chunk_id(self) -> str:
        return self.document.metadata.get("chunk_id", "?")

    @property
    def text(self) -> str:
        return self.document.page_content


@dataclass
class RAGAnswer:
    question: str
    answer: str
    contexts: list[RetrievedChunk]
    cited: list[int] = field(default_factory=list)  # 1-based indices into contexts
    latency_s: float = 0.0

    @property
    def abstained(self) -> bool:
        return is_abstention(self.answer)

    @property
    def cited_chunks(self) -> list[RetrievedChunk]:
        return [self.contexts[i - 1] for i in self.cited if 0 < i <= len(self.contexts)]


def is_abstention(answer: str) -> bool:
    norm = answer.strip().lower().replace("’", "'")
    return NO_ANSWER.lower().rstrip(".") in norm


def format_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[{i}] (source: {c.source})\n{c.text}" for i, c in enumerate(chunks, 1)
    )


class RAGPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        vectorstore: FAISS | None = None,
        llm: BaseChatModel | None = None,
        reranker: Reranker | None = None,
    ):
        self.settings = settings or get_settings()
        self.vectorstore = vectorstore or load_index(self.settings)
        self._llm = llm
        self._reranker = reranker

    @property
    def llm(self) -> BaseChatModel:
        if self._llm is None:
            self._llm = make_llm(self.settings.llm_model, self.settings.temperature)
        return self._llm

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = get_reranker(self.settings, llm=self._llm)
        return self._reranker

    # ---- retrieval -------------------------------------------------------------
    def vector_search(self, question: str, k: int | None = None) -> list[RetrievedChunk]:
        k = k or self.settings.fetch_k
        hits = self.vectorstore.similarity_search_with_score(question, k=k)
        # With normalised vectors FAISS returns squared L2 distance d = 2 - 2cos.
        return [
            RetrievedChunk(document=doc, vector_rank=i, vector_score=1.0 - float(dist) / 2.0)
            for i, (doc, dist) in enumerate(hits, 1)
        ]

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        top_k = top_k or self.settings.top_k
        candidates = self.vector_search(question, self.settings.fetch_k)
        by_id = {id(c.document): c for c in candidates}
        reranked = self.reranker.rerank(question, [c.document for c in candidates], top_k)
        out = []
        for doc, score in reranked:
            chunk = by_id[id(doc)]
            chunk.rerank_score = score
            out.append(chunk)
        return out

    # ---- generation ------------------------------------------------------------
    def generate(self, question: str, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return NO_ANSWER
        chain = ANSWER_PROMPT | self.llm | StrOutputParser()
        return chain.invoke({"question": question, "context": format_context(chunks)}).strip()

    def answer(self, question: str) -> RAGAnswer:
        start = time.perf_counter()
        chunks = self.retrieve(question)
        text = self.generate(question, chunks)
        cited = sorted({int(n) for n in _CITATION_RE.findall(text)})
        return RAGAnswer(
            question=question,
            answer=text,
            contexts=chunks,
            cited=cited,
            latency_s=time.perf_counter() - start,
        )
