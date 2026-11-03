"""Command-line interface.

    python -m rag_qa ingest                     # chunk + embed docs, save FAISS index
    python -m rag_qa ask "How many PTO days?"   # answer one question with citations
    python -m rag_qa chat                       # interactive loop
    python -m rag_qa eval                       # full retrieval + generation evaluation
    python -m rag_qa eval --retrieval-only --rerankers none cross-encoder
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import get_settings
from .pipeline import RAGAnswer, RAGPipeline


def _settings_from_args(args):
    return get_settings(
        embedding_provider=getattr(args, "embeddings", None),
        reranker=getattr(args, "reranker", None),
        llm_model=getattr(args, "model", None),
        top_k=getattr(args, "top_k", None),
        fetch_k=getattr(args, "fetch_k", None),
        chunk_size=getattr(args, "chunk_size", None),
        chunk_overlap=getattr(args, "chunk_overlap", None),
    )


def _print_answer(ans: RAGAnswer, show_context: bool) -> None:
    print(f"\n{ans.answer}\n")
    print("Sources:")
    for i, c in enumerate(ans.contexts, 1):
        mark = "*" if i in ans.cited else " "
        rr = f" rerank={c.rerank_score:.3f}" if c.rerank_score is not None else ""
        print(f" {mark}[{i}] {c.chunk_id}  ({c.document.metadata.get('section', '')})  "
              f"cos={c.vector_score:.3f}{rr}")
        if show_context:
            print("      " + c.text.replace("\n", "\n      "))
    print(f"\n(* = cited)  {ans.latency_s:.2f}s")


def cmd_ingest(args) -> None:
    from .vectorstore import build_index

    settings = _settings_from_args(args)
    _, chunks = build_index(settings, docs_path=args.docs)
    sources = sorted({c.metadata["source"] for c in chunks})
    print(f"Indexed {len(chunks)} chunks from {len(sources)} documents -> {settings.index_dir}")
    for s in sources:
        print(f"  - {s}: {sum(c.metadata['source'] == s for c in chunks)} chunks")


def cmd_ask(args) -> None:
    pipeline = RAGPipeline(_settings_from_args(args))
    _print_answer(pipeline.answer(args.question), args.show_context)


def cmd_chat(args) -> None:
    pipeline = RAGPipeline(_settings_from_args(args))
    print("Ask a question about the documents (empty line to quit).")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        _print_answer(pipeline.answer(q), args.show_context)


def cmd_eval(args) -> None:
    from .evaluate import run_evaluation, summary_dict

    settings = _settings_from_args(args)
    if args.rebuild:
        from .vectorstore import build_index

        store, _ = build_index(settings)
        pipeline = RAGPipeline(settings, vectorstore=store)
    else:
        pipeline = RAGPipeline(settings)

    report = run_evaluation(
        pipeline,
        qa_path=args.qa,
        rerankers=args.rerankers,
        with_generation=not args.retrieval_only,
        out_dir=args.out,
        progress=lambda m: print(f"\r  {m}   ", end="", file=sys.stderr, flush=True),
    )
    print(file=sys.stderr)
    print(json.dumps(summary_dict(report), indent=2))
    print(f"\nFull report: {args.out}/eval_report.md")


def build_parser() -> argparse.ArgumentParser:
    from .evaluate import DEFAULT_QA_PATH, RESULTS_DIR

    p = argparse.ArgumentParser(prog="rag_qa", description="Document Q&A with RAG")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp, retrieval=True):
        sp.add_argument("--embeddings", choices=["openai", "huggingface", "hashing"], help="embedding provider")
        if retrieval:
            sp.add_argument("--reranker", choices=["cross-encoder", "llm", "none"])
            sp.add_argument("--top-k", type=int)
            sp.add_argument("--fetch-k", type=int)
            sp.add_argument("--model", help="chat model for answering")

    sp = sub.add_parser("ingest", help="build the FAISS index")
    common(sp, retrieval=False)
    sp.add_argument("--docs", help="file or folder to ingest (default: data/docs)")
    sp.add_argument("--chunk-size", type=int)
    sp.add_argument("--chunk-overlap", type=int)
    sp.set_defaults(func=cmd_ingest)

    sp = sub.add_parser("ask", help="answer one question")
    common(sp)
    sp.add_argument("question")
    sp.add_argument("--show-context", action="store_true")
    sp.set_defaults(func=cmd_ask)

    sp = sub.add_parser("chat", help="interactive Q&A")
    common(sp)
    sp.add_argument("--show-context", action="store_true")
    sp.set_defaults(func=cmd_chat)

    sp = sub.add_parser("eval", help="evaluate retrieval + answers on the labeled set")
    common(sp)
    sp.add_argument("--qa", default=str(DEFAULT_QA_PATH))
    sp.add_argument("--out", default=str(RESULTS_DIR))
    sp.add_argument("--rerankers", nargs="+", choices=["cross-encoder", "llm", "none"],
                    help="re-rankers to compare in the retrieval eval (default: the configured one)")
    sp.add_argument("--retrieval-only", action="store_true", help="skip LLM answering/judging")
    sp.add_argument("--rebuild", action="store_true", help="rebuild the index before evaluating")
    sp.add_argument("--chunk-size", type=int)
    sp.add_argument("--chunk-overlap", type=int)
    sp.set_defaults(func=cmd_eval)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
