# Technology Guide — Document Q&A with RAG

This guide explains every technology, library, model, and technique in this project: **what it is, exactly where it is used (file and function), how it is used, and why it was chosen over the alternatives.** It also traces a question through the system step by step, explains the evaluation method and every metric formula, lists every configuration setting, and records the design trade-offs and known limitations.

If you only want to run the project, read [README.md](README.md). Read this file to understand the system well enough to explain, defend, or extend any part of it.

---

## Table of contents

1. [What the project does, in one paragraph](#1-what-the-project-does-in-one-paragraph)
2. [The big picture: three phases](#2-the-big-picture-three-phases)
3. [Where each technology lives (map of the codebase)](#3-where-each-technology-lives-map-of-the-codebase)
4. [Technologies in detail](#4-technologies-in-detail)
   - 4.1 [Python 3.11 and the standard library](#41-python-311-and-the-standard-library)
   - 4.2 [LangChain (core, text-splitters, community, openai)](#42-langchain)
   - 4.3 [FAISS — the vector index](#43-faiss--the-vector-index)
   - 4.4 [OpenAI API — embeddings and chat models](#44-openai-api--embeddings-and-chat-models)
   - 4.5 [sentence-transformers, PyTorch, Hugging Face Hub](#45-sentence-transformers-pytorch-and-hugging-face-hub)
   - 4.6 [Custom hashing embeddings](#46-custom-hashing-embeddings)
   - 4.7 [Pydantic — structured LLM output](#47-pydantic--structured-llm-output)
   - 4.8 [pypdf — PDF text extraction](#48-pypdf--pdf-text-extraction)
   - 4.9 [python-dotenv — configuration](#49-python-dotenv--configuration)
   - 4.10 [Streamlit — the web UI](#410-streamlit--the-web-ui)
   - 4.11 [pytest — the offline test suite](#411-pytest--the-offline-test-suite)
   - 4.12 [NumPy (indirect dependency)](#412-numpy-indirect-dependency)
5. [Core techniques in detail](#5-core-techniques-in-detail)
   - 5.1 [Document loading](#51-document-loading)
   - 5.2 [Structure-aware chunking](#52-structure-aware-chunking)
   - 5.3 [Embedding and indexing](#53-embedding-and-indexing)
   - 5.4 [Two-stage retrieval: vector search + re-ranking](#54-two-stage-retrieval-vector-search--re-ranking)
   - 5.5 [Prompt engineering against hallucination](#55-prompt-engineering-against-hallucination)
   - 5.6 [Citations and abstention detection](#56-citations-and-abstention-detection)
6. [Evaluation methodology in detail](#6-evaluation-methodology-in-detail)
7. [A question's full journey through the system](#7-a-questions-full-journey-through-the-system)
8. [Configuration reference](#8-configuration-reference)
9. [Measured results and how to read them](#9-measured-results-and-how-to-read-them)
10. [Design decisions and trade-offs](#10-design-decisions-and-trade-offs)
11. [Known limitations and how to extend the project](#11-known-limitations-and-how-to-extend-the-project)
12. [Explaining the project in an interview](#12-explaining-the-project-in-an-interview)
13. [Glossary](#13-glossary)

---

## 1. What the project does, in one paragraph

A user asks a question in natural language ("Can I carry unused vacation days into next year?"). The system searches a collection of documents for the passages most likely to contain the answer, hands only those passages to a large language model (LLM), and instructs the model to answer **using only those passages**, to **cite** which passage supports each statement, and to say **"I don't know based on the provided documents."** when the passages don't contain the answer. This pattern is called **Retrieval-Augmented Generation (RAG)**. It reduces *hallucination*, meaning confident answers the model makes up, because the model is grounded in real text rather than in its memory. The project also contains an **evaluation harness** that measures, against 48 hand-labeled questions, how good the search step is and how faithful the generated answers are.

---

## 2. The big picture: three phases

The system has three phases, and each one uses a different subset of the technologies.

### Phase A — Ingestion (offline, run once per document set)

```
data/docs/*.md|.txt|.pdf
        │  loaders.py        (pathlib, pypdf)
        ▼
LangChain Documents  (text + metadata: source file, page, format)
        │  chunking.py       (LangChain text splitters)
        ▼
~51 chunks of ≤800 characters, each tagged with its section path and chunk_id
        │  embeddings.py     (OpenAI / sentence-transformers / hashing)
        ▼
one vector per chunk  (1536 numbers for OpenAI, 384 for MiniLM, 1024 for hashing)
        │  vectorstore.py    (FAISS via langchain_community)
        ▼
index/index.faiss  +  index/index.pkl  +  index/manifest.json
```

Command: `python -m rag_qa ingest`

### Phase B — Question answering (online, per question)

```
question
   │ embed with the SAME embedding model used at ingestion
   ▼
FAISS nearest-neighbour search  →  20 candidate chunks (fetch_k)
   │ rerank.py: cross-encoder / LLM / none
   ▼
best 4 chunks (top_k)
   │ prompts.py: numbered context + grounding rules
   ▼
OpenAI chat model (gpt-4o-mini, temperature 0)
   │ pipeline.py: parse [n] citations, detect abstention
   ▼
RAGAnswer(answer, contexts, cited, latency)
```

Commands: `python -m rag_qa ask "..."`, `python -m rag_qa chat`, or `streamlit run app.py`

### Phase C — Evaluation (offline, per experiment)

```
data/eval/qa_set.jsonl (48 labeled questions)
   │
   ├── Retrieval eval: for each answerable question, compare vector-only ranking vs re-ranked
   │   ranking → Hit@k, Recall@k, MRR, nDCG@k, pool recall, per-category breakdown
   │
   └── Generation eval: run the full pipeline, then an LLM "judge" scores each answer
       → faithfulness, correctness, citation hit, abstention rate, false-refusal rate, latency
   ▼
results/eval_report.md  +  results/eval_report.json
```

Commands: `python -m rag_qa eval`, `python scripts/retrieval_sweep.py`

---

## 3. Where each technology lives (map of the codebase)

| File | Role | Technologies used there |
|---|---|---|
| [rag_qa/config.py](rag_qa/config.py) | All settings, read from environment / `.env` | python-dotenv, `dataclasses`, `os`, `pathlib` |
| [rag_qa/loaders.py](rag_qa/loaders.py) | Read `.md`, `.txt`, `.pdf` into Documents | pypdf, LangChain `Document`, `pathlib` |
| [rag_qa/chunking.py](rag_qa/chunking.py) | Split documents into chunks | `langchain_text_splitters` (`MarkdownHeaderTextSplitter`, `RecursiveCharacterTextSplitter`) |
| [rag_qa/embeddings.py](rag_qa/embeddings.py) | Turn text into vectors | `langchain_openai.OpenAIEmbeddings`, sentence-transformers, custom `HashingEmbeddings` (`hashlib`, `re`, `math`) |
| [rag_qa/vectorstore.py](rag_qa/vectorstore.py) | Build / save / load the index | FAISS through `langchain_community.vectorstores.FAISS`, `json` manifest |
| [rag_qa/rerank.py](rag_qa/rerank.py) | Second-stage re-ranking | sentence-transformers `CrossEncoder`, LangChain structured output, Pydantic |
| [rag_qa/prompts.py](rag_qa/prompts.py) | All prompt templates | LangChain `ChatPromptTemplate` |
| [rag_qa/pipeline.py](rag_qa/pipeline.py) | Retrieve → re-rank → generate | LangChain LCEL (`prompt \| llm \| parser`), `langchain_openai.ChatOpenAI`, `re`, `time` |
| [rag_qa/evaluate.py](rag_qa/evaluate.py) | Metrics, LLM judge, reports | Pydantic, LangChain structured output, `math`, `statistics`, `json` |
| [rag_qa/cli.py](rag_qa/cli.py) | Command-line interface | `argparse` |
| [app.py](app.py) | Web UI | Streamlit |
| [scripts/retrieval_sweep.py](scripts/retrieval_sweep.py) | Offline comparison of embeddings × re-ranker × k | everything above except OpenAI |
| [tests/](tests/) | 20 offline tests | pytest, LangChain `FakeListChatModel`, `RunnableLambda` |
| [data/docs/](data/docs/) | Sample corpus (6 fictional documents) | Markdown |
| [data/eval/qa_set.jsonl](data/eval/qa_set.jsonl) | 48 hand-labeled questions | JSON Lines |

Versions installed and tested in this project's `.venv`: Python 3.11.9, langchain-core 1.6.12, langchain-openai 1.7.2, langchain-community 0.4.2, langchain-text-splitters 1.1.3, faiss-cpu 1.15.1, openai 3.33.0, sentence-transformers 6.1.0, torch 2.14.1, transformers 5.20.1, huggingface_hub 1.33.0, pydantic 2.14.1, pypdf 6.21.0, python-dotenv 1.2.4, streamlit 1.66.0, pytest 9.1.1, numpy 2.4.6. `requirements.txt` sets lower bounds (`>=`), not exact pins, so a fresh install may get newer versions.

---

## 4. Technologies in detail

### 4.1 Python 3.11 and the standard library

**What it is.** The language the whole project is written in.

**Why Python.** Every major tool in the RAG/LLM ecosystem (LangChain, FAISS bindings, sentence-transformers, the OpenAI SDK, Streamlit) is Python-first. Python 3.11 is used because it is the installed interpreter and is comfortably supported by all of these libraries. It also gives clearer error messages and noticeably faster execution than 3.10 and earlier.

**Standard-library modules and why each is used:**

| Module | Where | Why |
|---|---|---|
| `dataclasses` | `config.Settings` (frozen), `pipeline.RetrievedChunk`, `pipeline.RAGAnswer`, `evaluate.QAItem` | Typed, self-documenting records with no boilerplate. `Settings` is `frozen=True` so settings can't be changed by accident mid-run; `with_overrides()` returns a modified *copy* using `dataclasses.replace`. |
| `pathlib` | everywhere paths appear | Cross-platform paths (this project was built on Windows; paths like `data/docs` must also work on macOS/Linux). |
| `argparse` | `cli.py` | Sub-commands (`ingest`, `ask`, `chat`, `eval`) with typed flags and `--help` for free, with no extra dependency such as Click or Typer. |
| `json` | `vectorstore.py` (manifest), `evaluate.py` (QA set, reports) | The QA set is **JSON Lines** (one JSON object per line), which is easy to append to, easy to diff in git, and readable by any tool. |
| `re` | `pipeline.py` (`_CITATION_RE = \[(\d+)\]`), `embeddings.py` (tokeniser), `evaluate.py` (whitespace normaliser) | Parsing citations out of the LLM answer, tokenising for hashing embeddings, collapsing whitespace before evidence matching. |
| `hashlib` | `embeddings.HashingEmbeddings` | MD5 used as a fast, *stable* hash function for feature hashing. Python's built-in `hash()` is randomised per process for strings, which would make vectors differ between runs. MD5 is not used for security here. |
| `math`, `statistics` | `embeddings.py` (vector norm), `evaluate.py` (log₂ for nDCG, `fmean` for averages) | Metric arithmetic. |
| `time` | `pipeline.answer` (`perf_counter` for latency), manifest timestamps | Latency is reported per answer and averaged in the eval. |
| `typing.Protocol` | `rerank.Reranker` | Defines the re-ranker interface ("anything with `name` and `rerank()`") without forcing inheritance. |
| `tempfile` | `app.py` | Uploaded files in the UI are written to a temporary folder before indexing. |

---

### 4.2 LangChain

**What it is.** LangChain is a framework of standard interfaces and integrations for LLM applications. It is split into several packages. This project uses four, and each has a specific job.

**Why LangChain at all.** RAG has many interchangeable parts: loaders, splitters, embedding models, vector stores, chat models, and prompts. LangChain gives each part a **common interface**, so you can swap one implementation for another without rewriting the pipeline. Concretely, this project switches between three embedding backends (OpenAI, local MiniLM, hashing) and between a real OpenAI model and a fake test model, and the pipeline code doesn't change. The trade-off is an extra dependency and some abstraction overhead. The project deliberately uses only LangChain's **stable building blocks** (documents, splitters, prompts, the pipe operator, the FAISS wrapper). It does *not* use high-level "agents" or pre-built "retrieval chains", so you can see and control every step.

#### 4.2.1 `langchain-core` — the base abstractions

| Abstraction | Where | How / why |
|---|---|---|
| `Document` (`langchain_core.documents`) | loaders, chunking, vector store, re-rankers | The universal unit: `page_content` (text) + `metadata` (dict). Metadata carries `source`, `page`, `format`, `section`, `chunk_id` through the entire pipeline, which is how citations can name the file a passage came from. |
| `Embeddings` (`langchain_core.embeddings`) | `embeddings.py` | The interface every embedding backend implements: `embed_documents(list[str])` and `embed_query(str)`. The custom `HashingEmbeddings` and `SentenceTransformerEmbeddings` classes subclass it, so FAISS accepts them exactly like OpenAI embeddings. |
| `BaseChatModel` (`langchain_core.language_models`) | type hints in pipeline, rerank, evaluate | Any chat model (OpenAI, a fake one in tests, or another provider later) plugs in. |
| `ChatPromptTemplate` (`langchain_core.prompts`) | `prompts.py` | Templates with `{context}` and `{question}` placeholders, separated into a **system** message (rules) and a **human** message (the question). Keeping rules in the system message makes the model treat them as instructions rather than as user content. |
| LCEL pipe operator `\|` | `pipeline.generate`: `ANSWER_PROMPT \| self.llm \| StrOutputParser()` | "LangChain Expression Language" composes steps into a *Runnable* chain: fill the template → call the model → extract the text. The same operator builds the re-rank chain and both judge chains. |
| `StrOutputParser` | `pipeline.generate` | Converts the model's message object into a plain string. |
| `.with_structured_output(PydanticModel)` | `rerank.LLMReranker`, `evaluate.Judge` | Makes the model return data that matches a Pydantic schema instead of free text (see 4.7). This removes fragile regex/JSON parsing of model output. |
| `RunnableLambda` | `tests/conftest.py` (`StructuredStub`) | Wraps a Python function as a Runnable, so tests can fake a structured-output model. |
| `FakeListChatModel` | tests | A chat model that returns pre-written replies, so tests of the full pipeline need no API key and are deterministic. |

#### 4.2.2 `langchain-text-splitters` — chunking

Used only in [chunking.py](rag_qa/chunking.py):

- **`MarkdownHeaderTextSplitter`** with `headers_to_split_on=[("#","h1"),("##","h2"),("###","h3")]` and `strip_headers=True`. It cuts a Markdown document at every heading and records the heading text in metadata (`h1`, `h2`, `h3`).
- **`RecursiveCharacterTextSplitter`** with `chunk_size=800`, `chunk_overlap=120`, and separators `["\n\n", "\n", ". ", " ", ""]`. It splits each section to fit the size budget, trying the "nicest" boundary first: paragraph, then line, then sentence, then word, and only as a last resort individual characters.

Why these two together is explained in detail in [5.2](#52-structure-aware-chunking).

#### 4.2.3 `langchain-community` — the FAISS wrapper

Used only in [vectorstore.py](rag_qa/vectorstore.py) (and its return type in pipeline.py): `from langchain_community.vectorstores import FAISS`. The wrapper:

- embeds documents with whichever `Embeddings` object you pass (`FAISS.from_documents(chunks, embeddings, normalize_L2=True)`),
- keeps a **docstore** mapping FAISS's integer vector IDs back to the original `Document` (text + metadata),
- provides `similarity_search_with_score(query, k)`,
- persists to disk with `save_local()` / `load_local()`.

**Note on its future:** at import time, langchain-community 0.4.2 prints a deprecation warning that the package is "being sunset". It works correctly today; the warning is filtered in `pytest.ini`. If it stops working in a future version, the fix is contained in one file (`vectorstore.py`): either move to a standalone FAISS integration package, or call `faiss` directly. The rest of the code only uses `similarity_search_with_score`, `from_documents`, `save_local` and `load_local`.

#### 4.2.4 `langchain-openai` — OpenAI integration

- **`OpenAIEmbeddings(model="text-embedding-3-small")`** in `embeddings.get_embeddings`.
- **`ChatOpenAI(model="gpt-4o-mini", temperature=0)`** in `pipeline.make_llm`. This one factory is used for the answering model, the LLM re-ranker, and the judge.

Both read `OPENAI_API_KEY` from the environment. The import is *lazy* (inside the function), so the project imports and its tests run even without the OpenAI package configured.

---

### 4.3 FAISS — the vector index

**What it is.** FAISS (Facebook AI Similarity Search, from Meta AI Research) is a C++ library with Python bindings for fast nearest-neighbour search over dense vectors. The `faiss-cpu` package is the CPU build.

**Where.** [vectorstore.py](rag_qa/vectorstore.py) (`build_index`, `load_index`) and [pipeline.py](rag_qa/pipeline.py) (`vector_search`).

**How it is used, step by step:**

1. **Index type.** The LangChain wrapper creates a `IndexFlatL2`, an *exact* (brute-force) index that compares the query with every stored vector. With ~51 chunks this takes microseconds. Approximate indexes such as IVF or HNSW only pay off at hundreds of thousands to millions of vectors, and they trade away a little accuracy, which is undesirable when we want to *measure* retrieval quality.
2. **Normalisation (`normalize_L2=True`).** Every vector is scaled to length 1 before being stored or queried. For unit vectors, squared Euclidean distance and cosine similarity are tied by an exact formula:

   ```
   ‖a − b‖² = ‖a‖² + ‖b‖² − 2·a·b = 1 + 1 − 2·cos(a,b) = 2 − 2·cos(a,b)
   ```

   So ranking by smallest L2 distance is *identical* to ranking by highest cosine similarity, which is the standard similarity measure for text embeddings. `pipeline.vector_search` converts the distance FAISS returns back into a readable cosine score: `cos = 1 − d / 2`. That is the `cos=0.612`-style number you see in the CLI and UI.
3. **Persistence.** `save_local()` writes `index.faiss` (the vectors) and `index.pkl` (the docstore, pickled). `load_local()` reads them back. `normalize_L2` is *not* saved by FAISS, so `load_index` passes it again. Without this, queries would not be normalised and scores would be wrong.
4. **`allow_dangerous_deserialization=True`.** Python pickle files can execute code when loaded, so LangChain refuses to load them unless you opt in. It is safe here because the project only loads an index it wrote itself. **Never load an `index.pkl` from an untrusted source.**
5. **Manifest guard.** `build_index` writes `manifest.json` (embedding provider and model, chunk size/overlap, number of documents and chunks, build time). `load_index` compares it with the current settings and raises a clear error on mismatch. This prevents a classic silent RAG bug: querying an index built with model A using vectors from model B. The dimensions might even match, but the similarities would be meaningless.

**Why FAISS over the alternatives:**

| Alternative | Why not here |
|---|---|
| Chroma, Qdrant, Weaviate, Milvus | Full databases (server or embedded DB) with metadata filtering, updates, multi-user access. Valuable in production, unnecessary for a single-user project over a few documents, and they add setup. |
| Pinecone and other managed services | Network calls, accounts, cost; the data leaves your machine. |
| A plain NumPy matrix + dot product | Would work at this size, but FAISS is the industry-standard library, scales to millions of vectors by changing the index type, and LangChain wraps it with a docstore and persistence for free. |

**Upgrade path:** if the corpus grows to hundreds of thousands of chunks, switch to `IndexHNSWFlat` or `IndexIVFFlat`, or move to a vector database. The `vectorstore.py` interface (`build_index` / `load_index` returning an object with `similarity_search_with_score`) is the only thing that needs to change.

---

### 4.4 OpenAI API — embeddings and chat models

**What it is.** The OpenAI API provides hosted embedding models and chat models. The `openai` Python SDK is the client library; `langchain-openai` wraps it.

**Where and which model:**

| Purpose | Model (default) | Setting | Code |
|---|---|---|---|
| Embedding chunks and questions | `text-embedding-3-small` (1536-dimensional vectors) | `RAG_EMBEDDING_MODEL` | `embeddings.get_embeddings` |
| Answering questions | `gpt-4o-mini` | `RAG_LLM_MODEL` | `pipeline.make_llm`, `RAGPipeline.generate` |
| LLM re-ranker (optional) | same as answering model | `RAG_RERANKER=llm` | `rerank.LLMReranker` |
| Judge for evaluation | `gpt-4o-mini` | `RAG_JUDGE_MODEL` | `evaluate.Judge` |

**Why these choices:**

- **`text-embedding-3-small`**: a strong general-purpose embedding model that is inexpensive and fast, with very good quality for English retrieval. Its vectors come normalised. A larger embedding model can be selected through `RAG_EMBEDDING_MODEL`, but on a corpus this small the gain is negligible.
- **`gpt-4o-mini`**: inexpensive, fast, and good at instruction following, which matters most here. A RAG answer is mostly "read four passages and restate the relevant facts with citations"; that doesn't need the largest model. Any OpenAI chat model name can be passed with `--model` or `RAG_LLM_MODEL`.
- **`temperature=0`**: makes output as deterministic as the API allows. We want the same question to get the same answer, both for users (consistency) and for evaluation (reproducibility). Creativity is undesirable when the goal is faithfulness to a source.
- **Separate judge model setting**: when you evaluate, it is good practice to make the judge at least as strong as the model being judged, and ideally a *different* model, to reduce "a model grading its own homework" bias. `RAG_JUDGE_MODEL` lets you set this independently.

**Why the project also works without OpenAI:** two local embedding backends (4.5, 4.6) and a local cross-encoder mean that ingestion, retrieval, re-ranking, and the full retrieval evaluation run with no API key. Only answer generation and LLM judging need the API.

---

### 4.5 sentence-transformers, PyTorch, and Hugging Face Hub

**What they are.**
- **sentence-transformers** is a library for running transformer models that produce sentence embeddings (bi-encoders) or pair scores (cross-encoders).
- **PyTorch** (`torch`) is the deep-learning runtime it uses. It runs on CPU here.
- **transformers** (Hugging Face) supplies the underlying model architectures and tokenisers.
- **Hugging Face Hub** is where the model weights are downloaded from on first use and then cached locally (in `~/.cache/huggingface`). Setting `HF_HUB_OFFLINE=1` afterwards skips the online update check.

These are used for two separate jobs.

#### Job 1 — the cross-encoder re-ranker (default re-ranker)

- **Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (setting `RAG_CROSS_ENCODER_MODEL`).
- **Where:** `rerank.CrossEncoderReranker`.
- **How:** for a question and 20 candidate chunks, it builds 20 pairs `(question, chunk_text)`, calls `model.predict(pairs)` to get one relevance score per pair, sorts by score, and keeps the top 4.
- **Why a cross-encoder:** see the bi-encoder vs cross-encoder explanation below.
- **Why this specific model:** it was trained on MS MARCO, a large dataset of real search queries with human-labeled relevant passages, which is exactly the "does this passage answer this question" task. It is small (6 transformer layers, MiniLM architecture), so it scores 20 pairs on a CPU in a fraction of a second, with no API cost and no data leaving the machine.

#### Job 2 — local embeddings (optional embedding backend)

- **Model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional vectors).
- **Where:** `embeddings.SentenceTransformerEmbeddings`, selected with `RAG_EMBEDDING_PROVIDER=huggingface` or `--embeddings huggingface`.
- **How:** `model.encode(texts, normalize_embeddings=True)`. Vectors are normalised for the same cosine reason as in 4.3.
- **Why:** a widely used, small, fast general-purpose embedding model that runs on CPU. It lets you build and evaluate a real semantic index with no API key. The project wraps it in a ~10-line class instead of adding the `langchain-huggingface` package, to keep dependencies down.

#### Bi-encoder vs cross-encoder — why there are two stages

| | Bi-encoder (embedding model + FAISS) | Cross-encoder (re-ranker) |
|---|---|---|
| How it works | Encodes the question and each passage **separately** into vectors; relevance = cosine of the two vectors | Reads question and passage **together** in one forward pass; outputs one relevance score |
| Can precompute? | Yes, passage vectors are computed once at ingestion | No, it needs the question, so it runs at query time |
| Speed | Searching millions of passages takes milliseconds | One model call per pair, so too slow for the whole corpus |
| Accuracy | Good, but coarse: the passage vector must summarise *everything* the passage is about, independent of the question | Higher: attention can directly match words and phrases in the question against the passage ("how far ahead" ↔ "at least 30 days in advance") |

The standard solution, used here, is **retrieve-then-rerank**: use the fast bi-encoder to shortlist `fetch_k=20` candidates, then use the accurate cross-encoder only on those 20 to pick the final `top_k=4`. This gets close to cross-encoder accuracy at close to bi-encoder speed. [Section 9](#9-measured-results-and-how-to-read-them) shows the measured gain.

**Practical note.** On the machine this was built on, importing `torch` from the project's drive took about 2 minutes per process. That's why [scripts/retrieval_sweep.py](scripts/retrieval_sweep.py) runs all configurations in **one** process instead of launching the CLI several times. It's also why the re-ranker and local embeddings are imported lazily (inside functions), so commands that don't need them stay fast.

---

### 4.6 Custom hashing embeddings

**What it is.** A small embedding model written from scratch in [embeddings.py](rag_qa/embeddings.py) (`HashingEmbeddings`, 1024 dimensions), using only the standard library.

**How it works:**
1. Lower-case the text, extract alphanumeric tokens with the regex `[a-z0-9]+`, and drop common stop-words ("the", "is", "how", …).
2. Add **bigrams** (adjacent word pairs such as `rate_limit`) to the unigrams, so some word order and phrase information survives.
3. For each token, compute MD5, take 8 bytes as an integer `h`. Use `h mod 1024` as the vector position, and the top bit of `h` to choose `+1` or `−1`. This is **signed feature hashing** (the "hashing trick"). The random signs make hash collisions tend to cancel out instead of piling up.
4. L2-normalise the vector.

**Why it exists:**
- **Tests and CI:** it is fully deterministic and needs no network, no model download, and no API key, so the 20 tests run in a few seconds anywhere.
- **A lexical baseline:** it only matches *shared words*, not meaning. Comparing it with MiniLM shows how much semantic embeddings help, and shows that the cross-encoder can rescue a weak first stage (Recall@1 0.663 → 0.884, see section 9).

**Limitation:** it doesn't understand synonyms or paraphrases ("vacation" vs "PTO"), so it is not intended for real use. That weakness is exactly why it's useful as a baseline.

---

### 4.7 Pydantic — structured LLM output

**What it is.** A data-validation library: you declare a class with typed fields, and Pydantic validates and converts data into it.

**Where and why:**

| Schema | File | Purpose |
|---|---|---|
| `_PassageScore(passage: int, score: int 0–3)`, `_RerankResult(scores: list)` | `rerank.py` | The LLM re-ranker must return one integer score per passage. `Field(ge=0, le=3)` enforces the range. |
| `_Claim(claim: str, supported: bool)`, `_FaithfulnessVerdict(claims: list)` | `evaluate.py` | The faithfulness judge must split the answer into claims and mark each as supported or not. |
| `_CorrectnessVerdict(reasoning: str, score: float)` | `evaluate.py` | The correctness judge must give a short reason and a score. Asking for the reasoning *before* the score tends to make the model's grading more careful. |

These classes are passed to LangChain's `llm.with_structured_output(Schema)`, which uses OpenAI's structured-output / function-calling support to make the model's reply conform to the schema and returns a Pydantic object. **Why:** without it you would ask the model to "reply in JSON" and then write fragile parsing code that breaks when the model adds a sentence before the JSON or a trailing comma. With it, the code reads `result.scores` or `verdict.claims` directly.

---

### 4.8 pypdf — PDF text extraction

**Where:** `loaders.load_file`, for `.pdf` files only.

**How:** `PdfReader(path)` and then `page.extract_text()` for each page. Each non-empty page becomes its own `Document` with `metadata={"source": filename, "page": n, "format": "pdf"}`, so answers can be traced to a page.

**Why pypdf:** pure Python (no system libraries to install), maintained, and sufficient for text-based PDFs. **Limitation:** scanned PDFs (images of text) contain no extractable text. They would need OCR (for example Tesseract) first. PDFs go through the recursive splitter only, because they have no Markdown headings.

---

### 4.9 python-dotenv — configuration

**Where:** `config.py` calls `load_dotenv()` at import time.

**Why:** secrets such as `OPENAI_API_KEY` and settings such as `RAG_TOP_K` live in a `.env` file that is **git-ignored** (see `.gitignore`), so keys never end up in the repository. `.env.example` documents every variable without real values. Real environment variables still take precedence over `.env`, which is how CI or deployment environments normally inject configuration.

---

### 4.10 Streamlit — the web UI

**Where:** [app.py](app.py). Run with `streamlit run app.py`.

**What the UI offers:**
- Sidebar: choose the embedding backend, re-ranker, `top_k`, `fetch_k`, and chat model. Enter an API key if none is set (it is kept only in that process's environment). Upload PDF/MD/TXT files. Build the index.
- Main area: a question box, the answer, latency, how many passages were cited, and an expandable view of every retrieved passage showing source file, section, vector rank, cosine score, re-rank score, and a ✅ for passages the answer cited.

**Streamlit features used and why:**
- `st.cache_resource` on `get_pipeline()`: loading the index and the cross-encoder is slow, and Streamlit re-runs the whole script on every interaction. Caching keeps one pipeline per settings combination. The settings tuple is passed as the cache key. The `Settings` object itself is passed as `_settings`, because Streamlit skips hashing arguments that start with an underscore.
- `st.cache_resource.clear()` after rebuilding the index, so the next question uses the new index.
- `st.file_uploader(accept_multiple_files=True)` + `tempfile.mkdtemp()` to index the user's own documents.

**Why Streamlit:** a complete interactive UI in under 100 lines of Python, with no HTML/JavaScript or separate backend. **Alternatives:** Gradio would be similar. FastAPI + a React frontend would be the production route, with an API other services can call, but it would be several times more code.

---

### 4.11 pytest — the offline test suite

**Where:** [tests/](tests/), configured by [pytest.ini](pytest.ini). Run with `python -m pytest -q` (20 tests, a few seconds).

**Design principle:** *no network, no API key, deterministic*. This is achieved with:
- `HashingEmbeddings` instead of OpenAI embeddings.
- `FakeListChatModel` that returns scripted answers.
- `StructuredStub` (in `conftest.py`): a tiny fake model whose `with_structured_output(schema)` returns a `RunnableLambda`. Tests can then feed exact re-rank scores or judge verdicts and check how the code handles them.
- Session-scoped fixtures (`settings`, `built_index`) that build the index **once** into a temporary folder and share it across tests.

**What is tested:**

| File | Tests |
|---|---|
| `test_data.py` | The labeled set is well-formed (unique IDs, answerable questions have evidence, at least 3 unanswerable). **Every evidence sentence appears verbatim in its source document.** **Every evidence sentence survives chunking intact**, meaning no chunk boundary cuts it in half; otherwise it could never be counted as retrieved. |
| `test_pipeline.py` | Loader reads all docs and rejects empty folders. Chunks respect the size limit, have unique IDs and carry section metadata. Overlap ≥ size is rejected. Hashing embeddings are normalised and deterministic. Index save/load round trip, and the manifest guard rejects a mismatched embedding model. Vector scores are valid cosines in descending order. Citations `[1][3]` are parsed and mapped to the right chunks. Abstention detection works, including a curly apostrophe. The prompt contains the grounding rules. The LLM re-ranker orders by score and breaks ties by original rank. |
| `test_evaluate.py` | Metric math on hand-computed cases: perfect ranking, late hit, cutoff at k, duplicate evidence not double counted. The generation eval computes faithfulness, correctness snapping, abstention and false-refusal rates correctly. The full retrieval-only eval runs and writes matching JSON and Markdown reports. |

**Why test the data itself:** in RAG projects the evaluation set is code too. A typo in an evidence string would silently make a question unanswerable for the retriever and drag every metric down. These tests catch that the moment the documents or labels are edited.

---

### 4.12 NumPy (indirect dependency)

NumPy is listed in `requirements.txt`, but the project's own code does **not** import it. It is a required dependency of FAISS and sentence-transformers, which use NumPy arrays for vectors and scores internally. It is listed explicitly so the environment is clearly reproducible.

---

## 5. Core techniques in detail

### 5.1 Document loading

[loaders.py](rag_qa/loaders.py) walks the docs folder (`rglob`, sorted for a deterministic order), keeps files with suffixes `.md`, `.markdown`, `.txt`, `.pdf`, and returns `Document` objects. Every document records `source` (file name) and `format` (`markdown`, `text`, or `pdf`). The format decides how it is chunked. An empty folder or a missing path raises a clear error instead of silently building an empty index.

### 5.2 Structure-aware chunking

**Why chunk at all?** Embedding a whole document into one vector blurs every topic in it together, so search becomes imprecise. LLM context is also limited and costs money per token. Small, focused chunks make retrieval precise and prompts short.

**Why not just fixed-size chunks?** Fixed windows cut through the middle of sections and sentences, mixing unrelated topics (the end of "Sick Leave" with the start of "Equipment"). That makes the chunk's vector a blend and confuses the LLM.

**The approach used** (`chunk_documents`):

1. **Split Markdown at headings** (`#`, `##`, `###`). A chunk never contains text from two sections.
2. **Split each section with the recursive splitter** (800 characters max, 120 overlap). It prefers paragraph breaks, then line breaks, then sentence ends, so pieces are coherent.
3. **Overlap of 120 characters** (15%): when a section must be split, the last ~120 characters of one chunk repeat at the start of the next. A fact sitting on a boundary is then fully present in at least one chunk. The test `test_every_evidence_span_survives_chunking` verifies this for all labeled evidence.
4. **Section path as metadata and as a text prefix.** Each chunk gets `metadata["section"] = "Lumora Systems Employee Handbook > 2. Paid Time Off"`, and its text starts with `[Lumora Systems Employee Handbook > 2. Paid Time Off]`. **Why the prefix:** a chunk saying "Unused PTO can be carried over…" doesn't mention *which* document or policy it belongs to. The heading adds that context to the embedding and to the cross-encoder's input, which improves matching for questions phrased around the topic.
5. **Stable chunk IDs** `"<file>::<n>"` (for example `employee_handbook.md::3`), used in citations, the UI, and evaluation reports.

**Why 800 / 120?** Around 150–200 tokens per chunk is large enough to hold a complete policy rule or table row with its context, and small enough that 4 chunks make a short, focused prompt. It also stays within the input limits of the local models (MiniLM bi-encoder ~256 tokens, cross-encoder 512). Both values are configurable (`RAG_CHUNK_SIZE`, `RAG_CHUNK_OVERLAP`). Because evaluation labels are evidence sentences rather than chunk IDs, you can run `python -m rag_qa eval --rebuild --chunk-size 500 --chunk-overlap 80` and compare configurations fairly on the same labels.

**Result on the sample corpus:** 51 chunks from 6 documents.

### 5.3 Embedding and indexing

`build_index` = load → chunk → `FAISS.from_documents(chunks, embeddings, normalize_L2=True)` → `save_local` → write manifest. The same embedding model **must** embed both the chunks (at ingestion) and the questions (at query time). Vectors from different models live in different "coordinate systems". The manifest guard (4.3) enforces this.

### 5.4 Two-stage retrieval: vector search + re-ranking

`RAGPipeline.retrieve(question)`:

1. **`vector_search(question, fetch_k=20)`**: FAISS returns the 20 nearest chunks. Each becomes a `RetrievedChunk` holding the document, its `vector_rank` (1–20) and `vector_score` (cosine).
2. **`reranker.rerank(question, candidates, top_k=4)`** returns the best 4 with a `rerank_score`.
3. The re-rank score is attached to each chunk, so the UI and CLI can show both the vector and re-rank score for transparency.

**Why `fetch_k=20` and `top_k=4`:** the candidate pool must be large enough that the right chunk is almost always *somewhere* in it. The eval reports this as **pool recall**, the ceiling any re-ranker can reach. But it must be small enough that scoring it is cheap. The final 4 keep the prompt short. Measurements (section 9) show that with re-ranking, `top_k=2` already reaches the same recall as `top_k=4` with MiniLM embeddings.

**The three re-rankers** (`rerank.get_reranker`):

| Name | How | When to use |
|---|---|---|
| `cross-encoder` (default) | MiniLM cross-encoder scores each pair locally | Best accuracy/cost balance; no API calls |
| `llm` | One chat-model call with all 20 numbered passages; the model returns a 0–3 score for each through structured output | When you want the strongest judgement and accept extra latency and cost per question. **Tie-breaking:** a 0–3 scale produces many ties, so each passage gets a tiny bonus `(n − i) / (10·n)` (always < 0.1) that prefers the passage FAISS ranked higher. This never overrides a real score difference. |
| `none` | Keeps FAISS order | Baseline for measuring how much re-ranking helps |

### 5.5 Prompt engineering against hallucination

The answer prompt in [prompts.py](rag_qa/prompts.py) has a **system message** with five numbered rules, followed by the numbered context passages, and a **human message** with the question. Each rule has a purpose:

| Rule | Why |
|---|---|
| 1. Use ONLY the context, not prior knowledge | The core grounding instruction. The sample corpus is fictional on purpose, so any "prior knowledge" would be invented. |
| 2. Cite passages as `[1]`, `[2][3]` | Makes every statement traceable, lets users verify answers, and lets the evaluation check that the cited passage actually contains the evidence. Passages are numbered in the prompt as `[1] (source: file.md)` so the model has a short handle to cite. |
| 3. If the answer isn't in the context, reply with **exactly** "I don't know based on the provided documents." | Gives the model an explicit, acceptable way out instead of guessing. Using a *fixed sentence* makes refusals detectable with a simple string check, with no judge needed (see 5.6). |
| 4. If only partially answered, answer the supported part and say what is missing | Prevents both extremes: refusing when half the answer is available, and filling the gap with invented details. |
| 5. Be concise (1–4 sentences unless a list is asked for) | Shorter answers contain fewer claims that could be unsupported, and they are easier to verify. |

Other prompt choices: rules live in the **system** role (the model treats them as instructions, not as user text). Each passage shows its **source file**. **Temperature 0** keeps output deterministic. If retrieval returns nothing, `generate()` returns the abstention sentence without calling the LLM at all.

### 5.6 Citations and abstention detection

- **Citations:** after generation, the regex `\[(\d+)\]` collects every cited number. `RAGAnswer.cited` stores them sorted and de-duplicated, and `cited_chunks` maps them back to the actual chunks (ignoring out-of-range numbers). The CLI marks cited sources with `*`; the UI marks them with ✅.
- **Abstention:** `is_abstention(answer)` lower-cases the answer, converts curly apostrophes (`’`) to straight ones (models often output "don’t"), and checks whether the fixed sentence is contained in it. This drives the abstention metrics in evaluation.

---

## 6. Evaluation methodology in detail

### 6.1 The labeled question set

[data/eval/qa_set.jsonl](data/eval/qa_set.jsonl) has **48 questions**, written by hand against the 6 sample documents. Each line looks like this:

```json
{"id": "hb-03", "category": "paraphrase",
 "question": "How far ahead do I need to request a two-week vacation?",
 "reference_answer": "Requests of five or more consecutive working days must be submitted at least 30 days in advance through the HR portal.",
 "answerable": true, "sources": ["employee_handbook.md"],
 "evidence": ["must be submitted at least 30 days in advance through the HR portal"]}
```

| Category | Count | What it tests |
|---|---|---|
| `factual` | 25 | Direct questions whose wording resembles the document |
| `paraphrase` | 16 | Same meaning, different words (for example "two-week vacation" vs "five or more consecutive working days"). This is where keyword search fails and semantic embeddings and cross-encoders earn their keep. |
| `multi-hop` | 2 | The answer needs facts from **two different documents** (for example warranty terms from the manual + the claim procedure from the support doc) |
| `unanswerable` | 5 | The documents do not contain the answer (price, CEO, a Go SDK, HomeKit, retirement match). The correct behaviour is to abstain. |

(43 answerable + 5 unanswerable = 48.)

**Why a fictional corpus?** If the documents were about a real product, the LLM might answer correctly *from memory* even when retrieval failed. Faithfulness and abstention numbers would then be inflated and meaningless. Fictional facts ("21 days of PTO", "E3 = temperature sensor fault") can only come from the retrieved text.

### 6.2 Evidence-span relevance labels (a key design decision)

Most RAG tutorials label relevance with chunk IDs ("question 7 is answered by chunk 12"). Those labels break as soon as you change chunk size or overlap, because chunk 12 no longer contains the same text. This project instead labels each question with **evidence spans**: short verbatim sentences from the source document. A retrieved chunk counts as relevant if it **contains** an evidence span, after normalising whitespace, case, and curly apostrophes (`evaluate.normalize`, `evaluate.evidence_hits`).

Benefits:
- Labels stay valid under **any** chunking configuration, so chunk sizes can be compared fairly.
- Multi-hop questions list several spans, which allows **partial** recall (1 of 2 facts found = 0.5).
- The test suite verifies all spans exist verbatim and survive chunking.

### 6.3 Retrieval metrics (computed in `retrieval_metrics`)

Only the 43 answerable questions are scored. Unanswerable questions have no relevant chunk by definition. For each question, the metrics are computed for the top **k** chunks, separately for the vector-only ranking and the re-ranked ranking, **over the same 20 candidates**, so the comparison isolates the effect of the re-ranker.

| Metric | Formula / definition | What it tells you |
|---|---|---|
| **Hit@k** | 1 if at least one evidence span appears in the top k chunks, else 0 | "Did the LLM get *anything* useful?" |
| **Recall@k** | (number of distinct evidence spans found in top k) ÷ (number of evidence spans) | "Did the LLM get *everything* it needs?" Differs from Hit@k only for multi-hop questions. |
| **MRR** (Mean Reciprocal Rank) | 1 ÷ (rank of the first relevant chunk); 0 if none in top k | How high the first useful chunk is placed. 1.0 = always first; 0.5 = typically second. Matters because LLMs pay most attention to early context. |
| **nDCG@k** | DCG = Σ over ranks r of (new spans found at r) ÷ log₂(r + 1); IDCG = the best possible DCG = Σ for r = 1…min(n, k) of 1 ÷ log₂(r + 1); nDCG = DCG ÷ IDCG | A rank-aware score in 0–1 that rewards finding all spans and finding them early. A span counts only the first time it's found, so duplicate chunks get no extra credit. |
| **Pool recall** | Recall over all 20 candidates | Ceiling for any re-ranker: if the evidence isn't in the pool, no re-ranker can find it. |

**Worked example** (from the tests): evidence = `["alpha beta", "gamma delta"]`, ranked chunks = `["nothing", "nope", "Alpha   BETA here"]`, k = 3 → Hit = 1, Recall = 1/2 = 0.5 (only one span found), MRR = 1/3 (first hit at rank 3), nDCG = (1/log₂4) ÷ (1/log₂2 + 1/log₂3) = 0.5 ÷ 1.631 ≈ 0.31.

Results are also broken down **per category**, so you can see, for example, that paraphrase questions benefit most from re-ranking.

### 6.4 Generation metrics (LLM-as-judge, in `evaluate_generation` and `Judge`)

For each of the 48 questions, the full pipeline produces an answer, and a judge LLM scores it:

| Metric | How it is computed | Why |
|---|---|---|
| **Faithfulness** | The judge splits the answer into individual factual claims (ignoring `[n]` markers) and marks each as supported or not supported **by the retrieved context**. Score = supported ÷ total claims. Averaged over answers that made claims (refusals are excluded). | The direct measure of hallucination: does the answer say anything the sources don't? Claim-level scoring is more precise than one yes/no per answer, because one invented detail in a four-sentence answer gives 0.75 rather than a blunt "fail". |
| **Fully-faithful rate** | Share of answers with faithfulness exactly 1.0 | How often an answer is *completely* clean |
| **Correctness** | The judge compares the answer with the human **reference answer**: 1.0 = all key facts and no contradiction, 0.5 = partial, 0.0 = wrong or no answer. The judge's number is snapped to the nearest of {0, 0.5, 1}. Answerable questions only. | Faithfulness alone isn't enough: an answer can be perfectly grounded in the *wrong* passage. Correctness checks it actually answers the question. Snapping keeps scores on a clean 3-level scale even if the judge outputs 0.9. |
| **Citation hit** | 1 if at least one *cited* chunk contains an evidence span | Checks that citations point to the passage that really supports the answer, not just any retrieved passage |
| **Unanswerable abstention rate** | Share of the 5 unanswerable questions where the model gave the fixed "I don't know" sentence | Higher is better. Measures resistance to making things up. |
| **False-refusal rate** | Share of answerable questions where the model abstained | Lower is better. The counter-metric: a model that always says "I don't know" would score 100% abstention but be useless. |
| **Mean latency** | Average seconds per question (retrieval + re-rank + generation) | Practical cost of each configuration |

The report also lists **questions to inspect**: every answer with an unsupported claim (quoting the claim), and every wrong abstention decision. This turns the numbers into concrete debugging leads.

**Limitations of LLM-as-judge:** judges can be lenient or inconsistent, and they may favour answers from their own model family. Mitigations used here: temperature 0, structured output, claim-level decomposition, a reference answer for correctness, and a separately configurable judge model. For a high-stakes system you would spot-check a sample of judge verdicts by hand.

### 6.5 Reports

`run_evaluation` writes `results/eval_report.json` (everything, including per-question rows) and `results/eval_report.md` (readable tables). `--rerankers none cross-encoder llm` runs the retrieval eval once per re-ranker, so they appear side by side. `scripts/retrieval_sweep.py` writes `results/retrieval_sweep.md`, comparing embeddings × re-ranker × k = 1, 2, 4 without needing an API key.

---

## 7. A question's full journey through the system

Example: `python -m rag_qa ask "How far ahead do I need to request a two-week vacation?"`

1. **CLI** (`cli.py`): argparse parses the `ask` command. `_settings_from_args` builds a `Settings` object (defaults ← `.env` ← CLI flags).
2. **Pipeline construction** (`RAGPipeline.__init__`): `load_index` checks that `index/index.faiss` exists, reads `manifest.json` and confirms the embedding model matches, then loads FAISS with `normalize_L2=True`. The LLM and re-ranker are created lazily on first use.
3. **Embedding the question**: FAISS calls `embed_query` on the configured embeddings (for example OpenAI `text-embedding-3-small`) → one normalised 1536-number vector.
4. **Vector search**: FAISS compares that vector with all 51 chunk vectors and returns the 20 closest. The PTO chunk may not be first: the question says "vacation" and "two-week", while the document says "PTO" and "five or more consecutive working days".
5. **Re-ranking**: the cross-encoder reads each `(question, chunk)` pair jointly. It recognises that "how far ahead … request" matches "must be submitted at least 30 days in advance", and moves the PTO chunk to position 1. The top 4 are kept.
6. **Prompt assembly**: `format_context` numbers the 4 chunks as `[1] (source: employee_handbook.md)` + text. `ANSWER_PROMPT` puts the rules and context in the system message and the question in the human message.
7. **Generation**: `ChatOpenAI(gpt-4o-mini, temperature=0)` returns something like: *"Requests for five or more consecutive working days must be submitted at least 30 days in advance through the HR portal [1]."*
8. **Post-processing**: the regex finds `[1]` → `cited=[1]`. `is_abstention` → False. Latency is measured.
9. **Output**: the CLI prints the answer, then all 4 sources with chunk ID, section path, cosine score, re-rank score, and `*` on cited ones.

In the **evaluation**, this same question (`hb-03`) is scored by checking whether a chunk containing *"must be submitted at least 30 days in advance through the HR portal"* appears in the top k (retrieval), and then by the judge (faithfulness against the 4 passages, correctness against the reference answer).

---

## 8. Configuration reference

Every setting can be set as an environment variable (or in `.env`). Many can also be overridden with CLI flags or in the Streamlit sidebar. Defined in [config.py](rag_qa/config.py).

| Variable | Default | CLI flag | Effect |
|---|---|---|---|
| `OPENAI_API_KEY` | — | — | Required for OpenAI embeddings, answering, LLM re-ranking, and judging |
| `RAG_DOCS_DIR` | `data/docs` | `ingest --docs` | Folder (or single file) to index |
| `RAG_INDEX_DIR` | `index` | — | Where the FAISS index and manifest are stored |
| `RAG_EMBEDDING_PROVIDER` | `openai` | `--embeddings` | `openai`, `huggingface` (local MiniLM), or `hashing` (lexical, offline) |
| `RAG_EMBEDDING_MODEL` | provider default | — | `text-embedding-3-small` / `sentence-transformers/all-MiniLM-L6-v2` / `hashing-1024` |
| `RAG_LLM_MODEL` | `gpt-4o-mini` | `--model` | Chat model for answers (and the LLM re-ranker) |
| `RAG_JUDGE_MODEL` | `gpt-4o-mini` | — | Chat model used as evaluation judge |
| `RAG_TEMPERATURE` | `0` | — | Sampling temperature for answers |
| `RAG_CHUNK_SIZE` | `800` | `--chunk-size` | Maximum characters per chunk (before the section prefix) |
| `RAG_CHUNK_OVERLAP` | `120` | `--chunk-overlap` | Characters repeated between neighbouring chunks; must be < chunk size |
| `RAG_RERANKER` | `cross-encoder` | `--reranker` | `cross-encoder`, `llm`, or `none` |
| `RAG_CROSS_ENCODER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | — | Any sentence-transformers cross-encoder |
| `RAG_FETCH_K` | `20` | `--fetch-k` | FAISS candidates passed to the re-ranker |
| `RAG_TOP_K` | `4` | `--top-k` | Chunks kept after re-ranking and sent to the LLM |

Changing the embedding provider/model or chunking requires re-running `ingest` (or `eval --rebuild`). The manifest guard will remind you if you forget to re-ingest after changing the embedding model.

---

## 9. Measured results and how to read them

Measured locally with `python scripts/retrieval_sweep.py` (no API key): 43 answerable questions, `fetch_k = 20`, cross-encoder `ms-marco-MiniLM-L-6-v2`.

| Embeddings | k | Vector Recall@k | Re-ranked Recall@k | Vector MRR | Re-ranked MRR | Vector nDCG@k | Re-ranked nDCG@k |
|---|---|---|---|---|---|---|---|
| hashing-1024 (lexical) | 1 | 0.663 | **0.884** | 0.674 | **0.907** | 0.674 | **0.907** |
| hashing-1024 (lexical) | 2 | 0.814 | **0.965** | 0.744 | **0.942** | 0.762 | **0.942** |
| hashing-1024 (lexical) | 4 | 0.895 | **0.965** | 0.769 | **0.942** | 0.800 | **0.942** |
| all-MiniLM-L6-v2 | 1 | 0.884 | **0.907** | 0.907 | **0.930** | 0.907 | **0.930** |
| all-MiniLM-L6-v2 | 2 | 0.965 | **0.988** | 0.942 | **0.965** | 0.942 | **0.965** |
| all-MiniLM-L6-v2 | 4 | 0.988 | 0.988 | 0.948 | **0.965** | 0.952 | **0.965** |

**How to interpret:**
1. **Semantic embeddings beat keyword matching.** At k = 1, MiniLM finds the evidence for 88% of questions vs 66% for hashing. Most of the gap is on paraphrased questions.
2. **Re-ranking helps most when the first stage is weak.** With lexical vectors, the cross-encoder lifts Recall@1 by 22 points (0.663 → 0.884) and MRR from 0.674 to 0.907. It finds the right chunk in the 20-candidate pool and moves it to the top.
3. **With good embeddings, re-ranking improves ordering more than coverage.** At k = 4 MiniLM already finds 98.8% of the evidence, so recall can't improve much. But MRR rises from 0.948 to 0.965: the right passage is first more often, and LLMs use early context most reliably.
4. **Re-ranking lets you send fewer chunks.** With re-ranking, k = 2 reaches the same recall (0.988) as k = 4. That halves the context, which means a cheaper, faster prompt with fewer distracting passages that could lead to unsupported claims.
5. **Small-corpus caveat.** With only 51 chunks, retrieval is relatively easy, and differences of one question equal 2.3 points (1/43). The *method* scales; on a larger or more confusing corpus, re-ranking typically matters more.

**Generation metrics** (faithfulness, correctness, abstention, citation hit) require an OpenAI key. Running `python -m rag_qa eval --rerankers none cross-encoder` produces them in `results/eval_report.md`. They were not run during development because no key was available on the build machine, so no generation numbers are claimed here.

---

## 10. Design decisions and trade-offs

| Decision | Alternative | Why this way |
|---|---|---|
| LangChain building blocks, not pre-built chains/agents | `RetrievalQA`-style high-level chains | Every step is visible, testable, and measurable separately. Pre-built chains hide retrieval scores and make re-ranking experiments harder. |
| FAISS flat (exact) index | HNSW/IVF, or a vector DB | Exact results at this size and zero infrastructure; evaluation isn't blurred by approximation error. |
| Header-aware + recursive chunking with section prefix | Fixed-size windows; semantic chunking with embeddings | Respects document structure cheaply and deterministically. Semantic chunking costs extra embedding calls and is harder to reason about. |
| Retrieve 20 → re-rank → keep 4 | Send top 4 straight from FAISS; or send 10+ chunks | Measured better ranking at a small CPU cost. Fewer chunks = cheaper prompts and less distraction. |
| Local cross-encoder as default re-ranker | LLM re-ranker | No per-query API cost or added network latency; strong quality from MS MARCO training. The LLM option is still available for comparison. |
| Fixed abstention sentence | Let the model phrase refusals freely | Refusals become machine-checkable, which enables abstention and false-refusal metrics without a judge. |
| Evidence-span labels | Chunk-ID labels | Labels survive re-chunking; partial credit for multi-hop. |
| Claim-level faithfulness judge | One yes/no "is this grounded?" | Finer-grained, and it names the exact unsupported claim for debugging. |
| Fictional corpus | Real public documents | Prevents the LLM answering from memory, so grounding metrics are honest. |
| Three embedding backends incl. hashing | OpenAI only | Tests and retrieval experiments run offline, deterministically, for free. |
| Manifest guard on the index | Trust the user | Catches the "wrong embedding model" bug, which otherwise fails silently. |
| Lazy imports of torch/OpenAI | Import everything at top | Commands and tests that don't need heavy libraries start fast (torch import took ~2 min on the build machine). |

---

## 11. Known limitations and how to extend the project

**Limitations:**
- **Corpus size:** 6 documents / 51 chunks. Retrieval scores are high partly because the corpus is small.
- **Evaluation set size:** 48 questions (only 2 multi-hop, 5 unanswerable). One question ≈ 2 percentage points, so small differences aren't statistically meaningful.
- **No incremental indexing:** adding a document means rebuilding the whole index (fast at this size).
- **Scanned PDFs** need OCR, which isn't included.
- **LLM-judge reliability:** judges can make mistakes. Generation metrics should be spot-checked by hand.
- **No conversation memory:** each question is answered independently. `chat` mode doesn't rewrite follow-up questions like "and what about part-time?" using the earlier turns.
- **langchain-community deprecation:** the FAISS wrapper package is marked as being sunset (section 4.2.3).
- **Pickle docstore:** only load indexes you built yourself.

**Natural extensions (where to make the change):**
1. **Hybrid search** (BM25 keyword + vector, merged with reciprocal-rank fusion) in `pipeline.vector_search`. This helps with exact codes such as "E3" or "PAK-2".
2. **Query rewriting / multi-query** before retrieval, for conversational follow-ups and vague questions.
3. **Metadata filtering** (for example only search `security_policy.md`) using FAISS's `filter=` argument.
4. **Streaming answers** in Streamlit using the chat model's `.stream()`.
5. **Bigger evaluation set**, including more multi-hop and "partially answerable" questions, plus confidence intervals (bootstrap) in the report.
6. **Production serving:** FastAPI endpoint around `RAGPipeline.answer`, a managed vector DB, response caching, and tracing/observability of each step.

---

## 12. Explaining the project in an interview

**30-second version:** "I built a RAG system that answers questions over a document set with citations. Documents are chunked along their heading structure, embedded with OpenAI embeddings, and stored in FAISS. At query time I retrieve 20 candidates and re-rank them with a cross-encoder down to 4, then a prompt template forces the model to answer only from those passages, cite them, and say 'I don't know' otherwise. I evaluated it on 48 hand-labeled questions: retrieval with Recall@k, MRR, and nDCG, and answers with an LLM judge for claim-level faithfulness, correctness, and abstention on unanswerable questions. Re-ranking raised Recall@1 from 0.66 to 0.88 with lexical embeddings and improved MRR with semantic embeddings, which let me halve the context sent to the LLM."

**Likely follow-up questions and short answers:**
- *Why re-rank if you already have embeddings?* Bi-encoders compress each passage into one vector without seeing the question. A cross-encoder reads both together and is more accurate, but too slow for the whole corpus, so it only runs on a shortlist. (Section 4.5.)
- *How do you know the answers aren't hallucinated?* Grounding prompt + citations + a fixed abstention sentence, measured with claim-level faithfulness and abstention on unanswerable questions, using a fictional corpus so the model can't answer from memory. (Sections 5.5, 6.4.)
- *How did you label relevance?* With verbatim evidence sentences instead of chunk IDs, so labels survive re-chunking and support partial credit for multi-hop. Tests verify every label. (Section 6.2.)
- *Why FAISS and not a vector database?* Exact search is instant at this scale and needs no infrastructure. The vector store is isolated in one module, so swapping in Qdrant/Pinecone later is local. (Section 4.3.)
- *What would you improve?* Hybrid BM25 + vector search for exact identifiers, query rewriting for conversational follow-ups, and a larger evaluation set with confidence intervals. (Section 11.)

---

## 13. Glossary

| Term | Meaning |
|---|---|
| **RAG** (Retrieval-Augmented Generation) | Retrieve relevant text first, then have an LLM generate an answer from it |
| **Hallucination** | A fluent, confident statement by an LLM that isn't supported by any source |
| **Chunk** | A small piece of a document (here ≤800 characters) that is embedded and retrieved as a unit |
| **Embedding** | A list of numbers (vector) representing the meaning of a text; similar meanings → nearby vectors |
| **Cosine similarity** | Similarity of two vectors by the angle between them; 1 = same direction |
| **Bi-encoder** | Model that embeds query and passage separately (used for vector search) |
| **Cross-encoder** | Model that scores a query–passage pair jointly (used for re-ranking) |
| **Vector index / store** | Data structure for finding the nearest vectors to a query vector (FAISS here) |
| **fetch_k / top_k** | Number of candidates from vector search / number kept after re-ranking |
| **Re-ranking** | Re-ordering retrieved candidates with a more accurate (slower) model |
| **Prompt template** | A prompt with placeholders (`{context}`, `{question}`) filled at run time |
| **System message** | The part of a chat prompt that sets rules and role; the model treats it as instructions |
| **Temperature** | Randomness of LLM output; 0 = most deterministic |
| **Structured output** | Making the LLM return data matching a schema (Pydantic class) instead of free text |
| **LCEL** | LangChain Expression Language: composing steps with `\|` |
| **Abstention** | The model declining to answer because the information isn't available |
| **Faithfulness** | Share of an answer's claims supported by the retrieved context |
| **Hit@k / Recall@k / MRR / nDCG** | Retrieval quality metrics (defined in section 6.3) |
| **Pool recall** | Recall over the whole candidate pool; the ceiling for any re-ranker |
| **LLM-as-judge** | Using an LLM to grade outputs against criteria or a reference |
| **Evidence span** | A verbatim sentence from a document that is needed to answer a labeled question |
| **Multi-hop question** | A question that needs facts from more than one passage or document |
| **Feature hashing** | Mapping tokens to vector positions with a hash function instead of a vocabulary |
| **Manifest** | Small JSON file recording how the index was built, used to block mismatched loading |
