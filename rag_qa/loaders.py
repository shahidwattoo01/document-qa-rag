"""Load raw documents (Markdown, plain text, PDF) into LangChain Documents."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".pdf"}


def load_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return [
            Document(page_content=text, metadata={"source": path.name, "page": i + 1, "format": "pdf"})
            for i, page in enumerate(reader.pages)
            if (text := (page.extract_text() or "").strip())
        ]
    if suffix in SUPPORTED_SUFFIXES:
        text = path.read_text(encoding="utf-8")
        fmt = "markdown" if suffix in {".md", ".markdown"} else "text"
        return [Document(page_content=text, metadata={"source": path.name, "format": fmt})]
    raise ValueError(f"Unsupported file type: {path}")


def iter_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield path


def load_documents(root: Path | str) -> list[Document]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Document path does not exist: {root}")
    docs: list[Document] = []
    for path in iter_files(root):
        docs.extend(load_file(path))
    if not docs:
        raise ValueError(f"No supported documents ({', '.join(sorted(SUPPORTED_SUFFIXES))}) found in {root}")
    return docs
