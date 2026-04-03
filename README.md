# RAG
# Local RAG System

**Retrieval-Augmented Generation over Private Documents**

Python 3.11+ · ChromaDB · Ollama / LLaMA 3 · nomic-embed-text-v1

---

## Author

Built as a personal project — fully local, zero cloud dependencies

---

## Purpose

Query private business documents (CSV, JSON, TXT) using natural language.

---

## Architecture

Embed → Store (ChromaDB) → Retrieve → Generate (LLaMA 3 via Ollama)

---

## Status

Working prototype — ingest and query pipelines complete

---

## Tech Stack

* Python
* sentence-transformers
* ChromaDB
* Ollama
* pandas
* torch

---

## 1. Project Overview

This project is a fully local Retrieval-Augmented Generation (RAG) system. It allows users to ask natural language questions over private datasets (support tickets, CRM records, sales notes, policy files) and receive accurate, grounded answers.

No data ever leaves the machine.

### What is RAG?

* **Retrieval**: Finds relevant document chunks using vector similarity
* **Generation**: Uses an LLM to produce answers grounded in retrieved context

This results in a system that behaves like a domain expert over your data.

---

## 2. System Architecture

The system consists of two independent pipelines sharing a ChromaDB vector store.

### 2.1 Ingestion Pipeline (`ingest.py`)

Processes raw data into embeddings.

**Steps:**

1. File discovery (`data/` folder)
2. Format parsing (CSV, JSON, TXT)
3. Chunking (overlapping segments)
4. Metadata tagging
5. Embedding (`nomic-embed-text-v1`)
6. Storage in ChromaDB

---

### 2.2 Query Pipeline (`query.py`)

Handles user questions interactively.

**Steps:**

1. Encode query (`search_query:` prefix)
2. Vector search (Top-K retrieval)
3. Deduplication
4. Prompt assembly
5. LLM generation (LLaMA 3 via Ollama)
6. Response with sources

---

## 3. Repository Structure

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

## 4. Chunking Strategy

### CSV Files

Multiple chunk types are generated:

* `schema`
* `preview`
* `column_stats`
* `row`
* `row_group`

This enables both granular lookup and aggregate reasoning.

---

### JSON Files

* Email threads → summary + messages + flattened structure
* Generic JSON → summary + flattened chunks

---

### TXT Files

* Chunked with overlap (1200 chars, 200 overlap)
* Split on paragraphs/sentences where possible

---

## 5. Embedding Model

**Model:** `nomic-ai/nomic-embed-text-v1`

Uses asymmetric prefixes:

* Documents → `search_document:`
* Queries → `search_query:`

Runs fully on CPU for stability.

---

## 6. Key Design Decisions

### Stable Chunk IDs

Deterministic IDs prevent duplicate ingestion.

### Retrieval Deduplication

Removes near-identical chunks before LLM input.

### Metadata Filtering

Supports query prefixes:

* `[ticket]`
* `[lead]`
* `[customer]`
* `[sales_note]`

### Fully Local

* No APIs
* No cloud
* All models run locally

---

## 7. Installation & Setup

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

### 4. Run ingestion

```bash
python ingest.py
```

### 5. Run query

```bash
python query.py
```

---

## 8. Configuration

### ingest.py

* `EMBED_BATCH_SIZE = 8`
* `CHROMA_UPSERT_BATCH = 100`
* `TXT_TARGET_CHARS = 1200`
* `TXT_OVERLAP_CHARS = 200`
* `CSV_ROW_GROUP_SIZE = 20`

### query.py

* `TOP_K = 10`
* `LLM_MODEL = "llama3"`

---

## 9. Limitations

* Aggregation limits on very large datasets
* No PDF support
* No conversational memory
* Limited by LLM capability
* No re-ranking step

---

## 10. Future Improvements

* PDF ingestion
* Cross-encoder re-ranking
* Streaming responses
* Web UI (Gradio / Streamlit)
* Hybrid search (BM25 + vector)
* Auto re-ingestion

---

## 11. Glossary

* **RAG**: Retrieval-Augmented Generation
* **Embedding**: Numeric representation of text
* **ChromaDB**: Vector database
* **Ollama**: Local LLM runtime
* **Chunk**: Text fragment stored as vector

---

## Summary

A fully local, production-style RAG pipeline designed for privacy, flexibility, and extensibility — ideal for querying sensitive business data without relying on external APIs.

---

📄 Source document: fileciteturn0file0
