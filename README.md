# Local RAG System

**Retrieval-Augmented Generation over Private Documents**

---

**Tech Stack:** Python 3.11+ · ChromaDB · Ollama / LLaMA 3 · nomic-embed-text-v1

---

## Overview

This project is a fully local Retrieval-Augmented Generation (RAG) system. It allows you to query private documents (CSV, JSON, TXT) using natural language and get grounded answers with source attribution.

No data ever leaves your machine.

### How it works

* **Retrieval:** Finds relevant document chunks using vector similarity
* **Generation:** Feeds those chunks into an LLM to generate accurate answers

---

## Architecture

The system has two independent pipelines:

### 1. Ingestion Pipeline (`ingest.py`)

Processes and stores your data.

| Stage          | What Happens                  | Key Detail                |
| -------------- | ----------------------------- | ------------------------- |
| File Discovery | Scans `data/` folder          | Supports CSV, JSON, TXT   |
| Parsing        | Uses format-specific loaders  | pandas / custom / raw     |
| Chunking       | Splits content into fragments | Overlap preserves context |
| Metadata       | Adds structured tags          | source_file, type, etc.   |
| Embedding      | Converts to vectors           | `nomic-embed-text-v1`     |
| Storage        | Saves to ChromaDB             | Deduplicated via IDs      |

---

### 2. Query Pipeline (`query.py`)

Handles user questions.

| Stage    | What Happens                | Key Detail              |
| -------- | --------------------------- | ----------------------- |
| Encode   | Adds `search_query:` prefix | Matches embedding space |
| Search   | Retrieves top-K chunks      | Optional filters        |
| Dedup    | Removes similar chunks      | Avoids redundancy       |
| Prompt   | Builds LLM input            | Context-only grounding  |
| Generate | Runs LLaMA 3 (Ollama)       | Fully local             |
| Output   | Returns answer + sources    | File paths included     |

---

## Repository Structure

```
project/
├── ingest.py
├── query.py
├── data/
│   ├── tickets/
│   ├── crm_records/
│   ├── sales_notes/
│   └── *.json / *.txt
└── chroma_store/
```

---

## Chunking Strategy

### CSV

| Type         | Purpose              |
| ------------ | -------------------- |
| schema       | File structure       |
| preview      | Sample rows          |
| column_stats | Aggregations         |
| row          | Single record lookup |
| row_group    | Multi-row analysis   |

Key idea: **precompute aggregates** to avoid scanning entire tables.

---

### JSON

* Email threads → summary + per-message chunks
* Generic JSON → flattened key-value paths

---

### TXT

* ~1200 chars per chunk
* 200-char overlap
* Splits prefer paragraphs/sentences

---

## Embedding Model

**Model:** `nomic-ai/nomic-embed-text-v1`

| Use Case  | Prefix             |
| --------- | ------------------ |
| Documents | `search_document:` |
| Queries   | `search_query:`    |

Important: Missing prefixes = worse retrieval.

---

## Key Design Decisions

### Stable Chunk IDs

Prevents duplicate ingestion.

```python
def stable_chunk_id(filepath, chunk_index, text):
    digest = hashlib.md5(text.encode()).hexdigest()[:10]
    return f"{rel_path}::chunk::{chunk_index}::{digest}"
```

---

### Deduplication

Removes overlapping chunks before LLM step.

---

### Metadata Filtering

Prefix queries to narrow scope:

| Prefix       | Filter          |
| ------------ | --------------- |
| [ticket]     | support tickets |
| [lead]       | CRM leads       |
| [customer]   | customers       |
| [sales_note] | sales notes     |

---

### Fully Local

* ChromaDB → local storage
* Embeddings → local model
* LLM → Ollama (LLaMA 3)

---

## Installation

### 1. Install dependencies

```bash
pip install chromadb sentence-transformers pandas torch ollama
```

### 2. Install Ollama

```bash
brew install ollama
ollama serve
ollama pull llama3
```

### 3. Add data

```bash
mkdir -p data/tickets data/crm_records data/sales_notes
```

---

### 4. Run ingestion

```bash
python ingest.py
```

---

### 5. Query

```bash
python query.py
```

Example:

```
[ticket] How many open tickets does Acme Corp have?
```

---

## Configuration

### ingest.py

| Setting             | Default | Purpose          |
| ------------------- | ------- | ---------------- |
| EMBED_BATCH_SIZE    | 8       | Memory control   |
| CHROMA_UPSERT_BATCH | 100     | Write batch size |
| TXT_TARGET_CHARS    | 1200    | Chunk size       |
| TXT_OVERLAP_CHARS   | 200     | Overlap          |
| CSV_ROW_GROUP_SIZE  | 20      | Rows per chunk   |

---

### query.py

| Setting   | Default |
| --------- | ------- |
| TOP_K     | 10      |
| LLM_MODEL | llama3  |

---

## Limitations

* Large aggregations may be incomplete
* No PDF support
* No conversation memory
* Limited by local LLM quality
* No re-ranking step

---

## Future Improvements

* PDF ingestion
* Re-ranking (cross-encoder)
* Streaming responses
* Web UI (Gradio/Streamlit)
* Hybrid search (BM25 + vectors)
* Auto re-ingestion

---

## Glossary

| Term      | Meaning                |
| --------- | ---------------------- |
| RAG       | Retrieval + generation |
| Embedding | Semantic vector        |
| ChromaDB  | Vector database        |
| Ollama    | Local LLM runtime      |
| Chunk     | Document fragment      |
| TOP_K     | Retrieved chunk count  |

---

**Fully local. No cloud. Built for private data.**
