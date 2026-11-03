"""Structure-aware chunking.

Markdown is first split on headers so chunks never straddle two sections, and each
chunk carries its section path in metadata (and as a short prefix in the text, which
helps both embedding and re-ranking). Every section is then split with a recursive
character splitter to respect the chunk-size budget.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


def _section_path(metadata: dict) -> str:
    return " > ".join(metadata[k] for k in ("h1", "h2", "h3") if metadata.get(k))


def chunk_documents(docs: list[Document], chunk_size: int = 800, chunk_overlap: int = 120) -> list[Document]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=_HEADERS, strip_headers=True)
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Document] = []
    counters: dict[str, int] = {}
    for doc in docs:
        if doc.metadata.get("format") == "markdown":
            sections = header_splitter.split_text(doc.page_content)
            for s in sections:
                s.metadata = {**doc.metadata, **s.metadata}
        else:
            sections = [doc]

        for section in sections:
            section_path = _section_path(section.metadata)
            for piece in char_splitter.split_text(section.page_content):
                piece = piece.strip()
                if not piece:
                    continue
                source = section.metadata["source"]
                idx = counters.get(source, 0)
                counters[source] = idx + 1
                metadata = {
                    k: v for k, v in section.metadata.items() if k not in ("h1", "h2", "h3")
                }
                metadata.update(chunk_id=f"{source}::{idx}", section=section_path)
                text = f"[{section_path}]\n{piece}" if section_path else piece
                chunks.append(Document(page_content=text, metadata=metadata))
    return chunks
