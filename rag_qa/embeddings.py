"""Embedding backends.

- ``openai``      : OpenAI embeddings (default, used for the real pipeline).
- ``huggingface`` : local sentence-transformers model, no API key needed.
- ``hashing``     : dependency-free hashed bag-of-words vectors. Lexical only, but fully
                    deterministic and offline, which makes it ideal for unit tests and CI.
"""

from __future__ import annotations

import hashlib
import math
import re

from langchain_core.embeddings import Embeddings

from .config import Settings

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by can do does for from has have how i in is it its my of on or "
    "the this to was what when where which who will with you your".split()
)


class HashingEmbeddings(Embeddings):
    """Signed feature-hashing of unigrams + bigrams, L2-normalised."""

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _tokens(self, text: str) -> list[str]:
        words = [w for w in _TOKEN_RE.findall(text.lower()) if w not in _STOPWORDS]
        return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in self._tokens(text):
            h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:8], "little")
            vec[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class SentenceTransformerEmbeddings(Embeddings):
    """Thin wrapper so we don't need the extra langchain-huggingface package."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True, batch_size=32).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()


def get_embeddings(settings: Settings) -> Embeddings:
    provider = settings.embedding_provider
    model = settings.resolved_embedding_model
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=model)
    if provider == "huggingface":
        return SentenceTransformerEmbeddings(model)
    if provider == "hashing":
        return HashingEmbeddings()
    raise ValueError(f"Unknown embedding provider: {provider!r} (expected openai | huggingface | hashing)")
