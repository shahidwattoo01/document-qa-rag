"""Streamlit UI:  streamlit run app.py"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st

from rag_qa.config import get_settings
from rag_qa.pipeline import RAGPipeline
from rag_qa.vectorstore import build_index, load_manifest

st.set_page_config(page_title="Document Q&A (RAG)", page_icon="📄", layout="wide")
st.title("📄 Document Q&A with RAG")
st.caption("LangChain · FAISS · OpenAI — answers are grounded in your documents, with citations.")

with st.sidebar:
    st.header("Settings")
    embeddings = st.selectbox("Embeddings", ["openai", "huggingface", "hashing"],
                              index=["openai", "huggingface", "hashing"].index(get_settings().embedding_provider))
    reranker = st.selectbox("Re-ranker", ["cross-encoder", "llm", "none"])
    top_k = st.slider("Chunks sent to the LLM (top_k)", 1, 10, 4)
    fetch_k = st.slider("FAISS candidates (fetch_k)", top_k, 50, 20)
    model = st.text_input("Chat model", get_settings().llm_model)
    if not os.getenv("OPENAI_API_KEY"):
        key = st.text_input("OpenAI API key", type="password")
        if key:
            os.environ["OPENAI_API_KEY"] = key

    settings = get_settings(embedding_provider=embeddings, reranker=reranker, top_k=top_k,
                            fetch_k=fetch_k, llm_model=model)

    st.divider()
    st.subheader("Index")
    uploads = st.file_uploader("Add your own documents", type=["pdf", "md", "txt"], accept_multiple_files=True)
    if st.button("Build index", type="primary"):
        with st.spinner("Chunking and embedding..."):
            if uploads:
                tmp = Path(tempfile.mkdtemp())
                for f in uploads:
                    (tmp / f.name).write_bytes(f.getvalue())
                build_index(settings, docs_path=tmp)
            else:
                build_index(settings)
        st.cache_resource.clear()
        st.success("Index built.")
    manifest = load_manifest(settings.index_dir)
    if manifest:
        st.caption(f"{manifest['num_chunks']} chunks from {manifest['num_documents']} docs · "
                   f"{manifest['embedding_provider']}/{manifest['embedding_model']}")


@st.cache_resource(show_spinner="Loading index...")
def get_pipeline(_settings, key: tuple) -> RAGPipeline:
    return RAGPipeline(_settings)


try:
    pipeline = get_pipeline(settings, (embeddings, reranker, top_k, fetch_k, model))
except (FileNotFoundError, ValueError) as e:
    st.info(f"{e}\n\nUse **Build index** in the sidebar.")
    st.stop()

question = st.text_input("Ask a question", placeholder="e.g. How many PTO days can I carry over?")
if question:
    with st.spinner("Retrieving and generating..."):
        ans = pipeline.answer(question)
    st.markdown(f"### Answer\n{ans.answer}")
    st.caption(f"{ans.latency_s:.2f}s · cited {len(ans.cited)} of {len(ans.contexts)} passages")
    st.markdown("### Retrieved passages")
    for i, c in enumerate(ans.contexts, 1):
        tag = "✅ cited" if i in ans.cited else ""
        rr = f" · rerank {c.rerank_score:.2f}" if c.rerank_score is not None else ""
        with st.expander(f"[{i}] {c.source} — {c.document.metadata.get('section', '')} "
                         f"(vector #{c.vector_rank}, cos {c.vector_score:.2f}{rr}) {tag}"):
            st.text(c.text)
