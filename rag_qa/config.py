"""Runtime configuration, read from environment variables (and .env if present)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    docs_dir: Path = field(default_factory=lambda: Path(_env("RAG_DOCS_DIR", str(PROJECT_ROOT / "data" / "docs"))))
    index_dir: Path = field(default_factory=lambda: Path(_env("RAG_INDEX_DIR", str(PROJECT_ROOT / "index"))))

    embedding_provider: str = field(default_factory=lambda: _env("RAG_EMBEDDING_PROVIDER", "google"))
    embedding_model: str = field(default_factory=lambda: _env("RAG_EMBEDDING_MODEL", ""))
    llm_model: str = field(default_factory=lambda: _env("RAG_LLM_MODEL", "gemini-1.5-flash"))
    judge_model: str = field(default_factory=lambda: _env("RAG_JUDGE_MODEL", "gemini-1.5-flash"))
    temperature: float = field(default_factory=lambda: float(_env("RAG_TEMPERATURE", "0")))

    chunk_size: int = field(default_factory=lambda: int(_env("RAG_CHUNK_SIZE", "800")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("RAG_CHUNK_OVERLAP", "120")))

    reranker: str = field(default_factory=lambda: _env("RAG_RERANKER", "cross-encoder"))
    cross_encoder_model: str = field(
        default_factory=lambda: _env("RAG_CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    )
    fetch_k: int = field(default_factory=lambda: int(_env("RAG_FETCH_K", "20")))
    top_k: int = field(default_factory=lambda: int(_env("RAG_TOP_K", "4")))

    def with_overrides(self, **kwargs) -> "Settings":
        """Return a copy with the given non-None fields replaced."""
        return replace(self, **{k: v for k, v in kwargs.items() if v is not None})

    @property
    def resolved_embedding_model(self) -> str:
        if self.embedding_model:
            return self.embedding_model
        return {
            "openai": "text-embedding-3-small",
            "google": "models/text-embedding-004",
            "huggingface": "sentence-transformers/all-MiniLM-L6-v2",
            "hashing": "hashing-1024",
        }.get(self.embedding_provider, "")


def get_settings(**overrides) -> Settings:
    return Settings().with_overrides(**overrides)
