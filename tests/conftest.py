"""Offline fixtures: hashing embeddings + fake chat models, so tests need no API key or network."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from langchain_core.runnables import RunnableLambda

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_qa.config import PROJECT_ROOT, get_settings  # noqa: E402
from rag_qa.embeddings import HashingEmbeddings  # noqa: E402
from rag_qa.vectorstore import build_index  # noqa: E402

DOCS_DIR = PROJECT_ROOT / "data" / "docs"


@pytest.fixture(scope="session")
def settings(tmp_path_factory):
    return get_settings(
        embedding_provider="hashing",
        reranker="none",
        index_dir=tmp_path_factory.mktemp("index"),
        docs_dir=DOCS_DIR,
    )


@pytest.fixture(scope="session")
def built_index(settings):
    store, chunks = build_index(settings, embeddings=HashingEmbeddings())
    return store, chunks


class StructuredStub:
    """Minimal stand-in for a chat model's ``with_structured_output``."""

    def __init__(self, fn):
        self.fn = fn

    def with_structured_output(self, schema):
        return RunnableLambda(lambda prompt_value: self.fn(schema, prompt_value))
