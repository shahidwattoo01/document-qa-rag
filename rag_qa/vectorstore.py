"""Build, persist, and load the FAISS index.

A small manifest is stored next to the index so we refuse to query an index with a
different embedding model than the one it was built with (a silent, common RAG bug).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from .chunking import chunk_documents
from .config import Settings
from .embeddings import get_embeddings
from .loaders import load_documents

MANIFEST = "manifest.json"


def build_index(
    settings: Settings,
    embeddings: Embeddings | None = None,
    docs_path: Path | str | None = None,
    save: bool = True,
) -> tuple[FAISS, list[Document]]:
    embeddings = embeddings or get_embeddings(settings)
    docs = load_documents(docs_path or settings.docs_dir)
    chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)
    # normalize_L2 makes L2 distance rank-equivalent to cosine similarity.
    store = FAISS.from_documents(chunks, embeddings, normalize_L2=True)
    if save:
        index_dir = Path(settings.index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        store.save_local(str(index_dir))
        manifest = {
            "embedding_provider": settings.embedding_provider,
            "embedding_model": settings.resolved_embedding_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "num_documents": len({d.metadata["source"] for d in docs}),
            "num_chunks": len(chunks),
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        (index_dir / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return store, chunks


def load_manifest(index_dir: Path | str) -> dict:
    path = Path(index_dir) / MANIFEST
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_index(settings: Settings, embeddings: Embeddings | None = None) -> FAISS:
    index_dir = Path(settings.index_dir)
    if not (index_dir / "index.faiss").exists():
        raise FileNotFoundError(f"No FAISS index in {index_dir}. Run `python -m rag_qa ingest` first.")
    manifest = load_manifest(index_dir)
    expected = (settings.embedding_provider, settings.resolved_embedding_model)
    found = (manifest.get("embedding_provider"), manifest.get("embedding_model"))
    if manifest and found != expected:
        raise ValueError(
            f"Index at {index_dir} was built with {found[0]}/{found[1]} but settings use "
            f"{expected[0]}/{expected[1]}. Re-run ingest or change RAG_EMBEDDING_PROVIDER."
        )
    embeddings = embeddings or get_embeddings(settings)
    # The docstore is a pickle we wrote ourselves, so deserialising it is safe here.
    return FAISS.load_local(
        str(index_dir), embeddings, allow_dangerous_deserialization=True, normalize_L2=True
    )
