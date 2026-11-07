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
    if model.startswith("gemini"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, temperature=temperature)
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=model, temperature=temperature)


@dataclass
class RetrievedChunk:
    document: Document
    vector_rank: int
    vector_score: float
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
    cited: list[int] = field(default_factory=list)
    latency_s: float = 0.0

    @property
    def abstained(self) -> bool:
        return is_abstention(self.answer)
