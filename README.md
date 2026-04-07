# Local RAG System

**Retrieval-Augmented Generation over Structured Private Data**

**Tech Stack:** Python 3.11+ · ChromaDB · Ollama (LLaMA 3) · nomic-embed-text-v1

---

## Overview

This project implements a fully local Retrieval-Augmented Generation (RAG) system designed for querying structured business data (CRM records, support tickets, sales notes, emails).

The focus is not just on retrieval, but on **data preparation, chunk design, and filtering**, which are the primary failure points in naive RAG systems.

In practice, RAG quality is dominated by data quality—not model choice.
This system is built to address:

* noisy and inconsistent raw data
* duplicated semantic content
* lack of structured filtering
* unstable retrieval results

---

## System Architecture

Two independent pipelines:

### 1. Ingestion Pipeline (`ingest.py`)

Processes raw data into embeddings stored in ChromaDB.

### 2. Query Pipeline (`query.py`)

Handles user queries and generates grounded responses using retrieved context.

---

### Query Flow

```
User Query
   ↓
Encode (search_query:)
   ↓
Vector Search (ChromaDB)
   ↓
Metadata Filtering
   ↓
Deduplication
   ↓
Prompt Construction
   ↓
LLM (LLaMA 3 via Ollama)
   ↓
Answer + Sources
```

---

## Key Design Decisions

### 1. Two-stage data pipeline

**Design:**

* Stage 1 → data cleaning & normalization
* Stage 2 → RAG-specific transformation

**Why:**

* Keeps clean data reusable for analytics
* Allows iteration on retrieval without reprocessing raw data

**Tradeoff:**

* Increased pipeline complexity

---

### 2. Structured chunks over free text

Chunks are formatted as labeled key-value pairs.

**Why:**

* Improves embedding consistency
* Makes outputs interpretable when surfaced

**Tradeoff:**

* Less natural language → potentially weaker generative fluency

---

### 3. Metadata filtering before retrieval

Queries can include filters (e.g. `[ticket]`, priority, sales rep).

**Why:**

* Reduces search space
* Improves precision vs pure vector similarity

**Tradeoff:**

* Requires structured metadata upfront

---

### 4. Duplicate content handling via flags

Repeated content is **flagged, not removed**.

**Why:**

* Prevents vector collapse (identical embeddings dominating results)
* Preserves full dataset for auditability

**Tradeoff:**

* Requires careful filtering to avoid noisy retrieval

---

### 5. Dataset-specific chunk strategies

Example (support tickets):

* `rag_chunk_full` → includes resolution
* `rag_chunk_issue` → excludes resolution

**Why:**

* Prevents resolution text from biasing similarity search

**Tradeoff:**

* Increased storage and ingestion complexity

---

### 6. Ordinal feature encoding

Fields like priority and budget are mapped to numeric ranks.

**Why:**

* Enables range-based filtering in vector DB

**Tradeoff:**

* Requires additional preprocessing logic

---

## Data Pipeline

### Stage 1 — Cleaning (`data_cleaning.py`)

Goal: **correctness and consistency**

* column normalization (naming, casing, formatting)
* null handling and whitespace cleanup
* email / phone validation
* categorical standardization (status, source, priority)
* deduplication (row-level + ID-level)
* derived features (risk scores, flags, metrics)

Output: clean, structured datasets usable beyond RAG

---

### Stage 2 — RAG Transformation

Goal: **retrieval quality**

* rebuild text into structured `rag_chunk`
* attach metadata for filtering
* remove zero-signal columns
* flag low-quality or recycled content
* enrich data (topics, buckets, derived categories)

Output:

* `rag_chunk` → embedded
* `metadata` → stored for filtering

---

## Chunking Strategy

### CSV

* schema chunks (structure awareness)
* preview chunks (sample rows)
* row chunks (record lookup)
* row_group chunks (aggregations)

**Key idea:** precompute aggregates instead of scanning entire tables

---

### JSON

* email threads → summary + per-message chunks
* generic JSON → flattened key-value paths

---

### TXT

* ~1200 characters per chunk
* 200-character overlap
* paragraph-aware splitting

---

## Embedding Strategy

**Model:** `nomic-ai/nomic-embed-text-v1`

| Use Case  | Prefix             |
| --------- | ------------------ |
| Documents | `search_document:` |
| Queries   | `search_query:`    |

**Reasoning:**
Aligns query and document embeddings in the same vector space.

---

## Retrieval Pipeline

* top-K similarity search
* optional metadata filtering
* deduplication of overlapping chunks
* context-only prompt construction

---

## Example

```
[ticket] How many open tickets does Acme Corp have?
```

**Output:**

```
Acme Corp has 12 open tickets.

Sources:
- data/tickets/ticket_123.csv
- data/tickets/ticket_456.csv
```

---

## Limitations

* Retrieval degrades for vague or underspecified queries
* Large aggregations may be incomplete (chunk-based retrieval)
* No hybrid search (BM25 + vectors)
* No conversation memory
* Limited by local LLM performance
* No PDF ingestion

---

## Future Improvements

* hybrid search (BM25 + vector)
* re-ranking (cross-encoder)
* PDF ingestion
* streaming responses
* automated re-ingestion


