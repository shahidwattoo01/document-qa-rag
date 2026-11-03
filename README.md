# Document Q&A with RAG

A Retrieval-Augmented Generation pipeline that answers questions about a document collection, with citations, and an evaluation harness that measures retrieval quality and answer faithfulness against a hand-labeled question set.

**Stack:** Python · LangChain · FAISS · OpenAI API · sentence-transformers (cross-encoder re-ranking) · Streamlit

> For every technology, where it is used and why, plus the full evaluation method, see **[TECHNOLOGY_GUIDE.md](TECHNOLOGY_GUIDE.md)**.

```
                ┌──────────── ingest ────────────┐
 docs (md/txt/pdf) → header-aware chunking → embeddings → FAISS index (+ manifest)
                                                              │
 question ──► embed ──► FAISS top fetch_k (20) ──► re-ranker ──► top_k (4) ──► prompt template ──► LLM
                                                  (cross-encoder                (grounding rules,      │
                                                   | LLM | none)                 citations, abstain)   ▼
                                                                                         answer + [n] citations
```

## Features

- **Chunking & embedding** — Markdown is split on headers first so chunks never straddle sections, then by a recursive character splitter (800 chars / 120 overlap by default). Each chunk carries its section path (`Handbook > 2. Paid Time Off`) in metadata and as a short prefix that helps both embedding and re-ranking. PDF and plain text are supported too.
- **FAISS vector store** — L2-normalised vectors (cosine ranking), persisted to `index/` with a manifest; loading an index with a different embedding model than it was built with fails loudly instead of returning garbage.
- **Two-stage retrieval with re-ranking** — FAISS retrieves a wide candidate pool (`fetch_k=20`), then a re-ranker scores each (question, passage) pair jointly and keeps the best `top_k=4`:
  - `cross-encoder` — `cross-encoder/ms-marco-MiniLM-L-6-v2`, local, CPU, no API cost (default)
  - `llm` — the chat model grades each passage 0–3 via structured output
  - `none` — vector order (baseline)
- **Prompt templates that reduce hallucination** (`rag_qa/prompts.py`) — answer only from numbered context, cite `[n]` for every statement, answer partially when only partly supported, and reply with a fixed abstention sentence when the answer isn't in the documents (which makes refusals machine-checkable).
- **Evaluation harness** (`rag_qa/evaluate.py`) against `data/eval/qa_set.jsonl` — 48 hand-labeled questions (factual, paraphrased, multi-hop across two documents, and unanswerable):
  - Retrieval: Hit@k, Recall@k, MRR, nDCG@k for vector-only vs. re-ranked, plus candidate-pool recall (the ceiling any re-ranker can reach), broken down by category.
  - Generation (LLM-as-judge): claim-level **faithfulness** (supported claims / claims), **correctness** vs. the human reference, **citation hit rate**, **abstention rate** on unanswerable questions and **false-refusal rate** on answerable ones.
  - Relevance labels are *verbatim evidence spans*, not chunk IDs, so they stay valid when you change chunk size — you can compare chunking configs on the same labels.
- **CLI** and **Streamlit UI** (upload your own PDFs, inspect retrieved passages with vector/re-rank scores and which ones were cited).
- **Offline test suite** — 20 tests using deterministic hashing embeddings and fake chat models; no API key or network needed.

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env          # then put your OPENAI_API_KEY in .env
```

```bash
python -m rag_qa ingest
```

```bash
python -m rag_qa ask "Can I carry unused vacation days into next year?" --show-context
```

```bash
streamlit run app.py
```

Example output (illustrative, exact wording and scores vary):

```
Yes. Unused PTO can be carried over into the next calendar year, up to a maximum of 5 days,
and carried-over days must be used by March 31 or they expire [1].

Sources:
 *[1] employee_handbook.md::3  (Lumora Systems Employee Handbook > 2. Paid Time Off)  cos=0.612 rerank=7.914
  [2] employee_handbook.md::2  (...)
```

### No OpenAI key?

Retrieval runs fully locally with `--embeddings huggingface` (sentence-transformers `all-MiniLM-L6-v2`) or `--embeddings hashing`, and the cross-encoder re-ranker is local too:

```bash
python -m rag_qa ingest --embeddings huggingface
```

```bash
python -m rag_qa eval --embeddings huggingface --retrieval-only --rerankers none cross-encoder
```

## Evaluation

```bash
python -m rag_qa eval --rerankers none cross-encoder llm
```

Writes `results/eval_report.md` and `results/eval_report.json` (per-question rows, unsupported claims, wrong abstentions). Useful flags: `--retrieval-only`, `--rebuild --chunk-size 500 --chunk-overlap 80`, `--top-k`, `--fetch-k`, `--model`.

### Results

Retrieval, measured locally (`python scripts/retrieval_sweep.py`, no API key needed). There are 43 answerable questions, fetch_k = 20, and the cross-encoder re-ranks the FAISS candidates:

| Embeddings | k | Vector Recall@k | + Cross-encoder Recall@k | Vector MRR | + Cross-encoder MRR |
|---|---|---|---|---|---|
| hashing (lexical baseline) | 1 | 0.663 | **0.884** | 0.674 | **0.907** |
| hashing (lexical baseline) | 2 | 0.814 | **0.965** | 0.744 | **0.942** |
| hashing (lexical baseline) | 4 | 0.895 | **0.965** | 0.769 | **0.942** |
| all-MiniLM-L6-v2 | 1 | 0.884 | **0.907** | 0.907 | **0.930** |
| all-MiniLM-L6-v2 | 2 | 0.965 | **0.988** | 0.942 | **0.965** |
| all-MiniLM-L6-v2 | 4 | 0.988 | 0.988 | 0.948 | **0.965** |

What the numbers show:
- Re-ranking pays off most when the first stage is weak, and when you can only send a few chunks. At k = 1 with lexical vectors it adds +22 points of recall.
- With dense embeddings the first stage already finds the evidence at k = 4. The cross-encoder still moves the right chunk higher (MRR 0.948 → 0.965), so the LLM sees the answer first and `top_k` can be cut to 2 with no loss in recall. That means a shorter prompt and less distracting context.

The generation metrics (faithfulness, correctness, abstention, citations) need an `OPENAI_API_KEY`. Run `python -m rag_qa eval --rerankers none cross-encoder`, and the table is written to `results/eval_report.md`.

## Project layout

```
rag_qa/
  config.py       settings from env / .env
  loaders.py      md / txt / pdf -> Documents
  chunking.py     header-aware + recursive chunking
  embeddings.py   openai | huggingface | hashing
  vectorstore.py  build / save / load FAISS (+ manifest guard)
  rerank.py       cross-encoder, LLM, and no-op re-rankers
  prompts.py      answer, re-rank, and judge prompt templates
  pipeline.py     RAGPipeline: retrieve -> re-rank -> generate, citations, abstention
  evaluate.py     retrieval + generation metrics, Markdown/JSON reports
  cli.py          ingest | ask | chat | eval
scripts/retrieval_sweep.py   embeddings x re-ranker x k comparison (offline)
app.py            Streamlit UI
data/docs/        sample corpus (fictional company "Lumora Systems", 6 documents)
data/eval/        hand-labeled question set (48 questions)
tests/            offline pytest suite
TECHNOLOGY_GUIDE.md  detailed explanation of every technology and design decision
```

The sample corpus is fictional on purpose: the model can't answer from its pre-training knowledge, so faithfulness and abstention scores actually measure grounding.

## Testing

```bash
python -m pytest -q
```

The tests also validate the labeled set itself: every evidence span must appear verbatim in its source document and survive chunking intact.

## Configuration

All settings can be set via environment variables (see `.env.example`) or CLI flags: `RAG_EMBEDDING_PROVIDER`, `RAG_EMBEDDING_MODEL`, `RAG_LLM_MODEL` (default `gpt-4o-mini`), `RAG_JUDGE_MODEL`, `RAG_RERANKER`, `RAG_CHUNK_SIZE`, `RAG_CHUNK_OVERLAP`, `RAG_FETCH_K`, `RAG_TOP_K`, `RAG_DOCS_DIR`, `RAG_INDEX_DIR`.
