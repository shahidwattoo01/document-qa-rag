"""Sanity checks on the hand-labeled evaluation set."""

from rag_qa.evaluate import evidence_hits, load_qa_set, normalize
from rag_qa.loaders import load_documents

from .conftest import DOCS_DIR


def test_qa_ids_unique_and_well_formed():
    items = load_qa_set()
    assert len(items) >= 40
    assert len({q.id for q in items}) == len(items)
    for q in items:
        assert q.question.strip() and q.reference_answer.strip()
        if q.answerable:
            assert q.evidence and q.sources, q.id
        else:
            assert not q.evidence, q.id
    assert sum(not q.answerable for q in items) >= 3


def test_every_evidence_span_is_verbatim_in_its_sources():
    docs = {d.metadata["source"]: normalize(d.page_content) for d in load_documents(DOCS_DIR)}
    for q in load_qa_set():
        for ev in q.evidence:
            assert any(normalize(ev) in docs[s] for s in q.sources), f"{q.id}: {ev!r} not found"


def test_every_evidence_span_survives_chunking(built_index):
    _, chunks = built_index
    for q in load_qa_set():
        for i, _ in enumerate(q.evidence):
            assert any(i in evidence_hits(c.page_content, q.evidence) for c in chunks), (
                f"{q.id}: evidence #{i} is split across chunks"
            )
