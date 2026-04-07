# Local RAG System

**Retrieval-Augmented Generation over Private Documents**

---

**Tech Stack:** Python 3.11+ · ChromaDB · Ollama / LLaMA 3 · nomic-embed-text-v1

---

## Overview

This project is a fully local Retrieval-Augmented Generation (RAG) system. It allows you to query private documents (CSV, JSON, TXT) using natural language and get grounded answers with source attribution.


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
## Data handling 

Here’s a clean, merged README section in your style, with the pipeline distinction woven in naturally instead of feeling bolted on:

---

Before actually doing any retrieval, I cleaned the data.

I was given N files — 1 JSON with emails sent, 4 CSVs with mainly customer information, and a set of TXT files spanning different domains, from FAQs to company policies.

The cleaning happens in **two stages**, and that distinction matters because each stage solves a different problem.

---

### Stage 1 — raw → structured and clean (`data_cleaning.py`)

This is the first pass over all 4 CSVs (customers, leads, sales notes, support tickets). The goal here is **correctness and consistency**, not retrieval.

Specifically:

**Column name standardisation.**
Strips whitespace, lowercases, replaces special characters with underscores. This is defensive — raw CSVs break pipelines in stupid ways (trailing spaces, slashes in names), so this removes that risk upfront.

**Whitespace and null handling.**
All text fields are stripped, and blank strings (`" "`) are converted to `pd.NA`. Without this, `notna()` checks silently fail later.

**Field-level normalisation.**

* Emails lowercased and validated (invalid → null)
* Phones converted to digits-only with leading `+` preserved
* Company names split into three versions:

  * `company_name_raw` (audit trail)
  * `company_name_normalized` (lowercased, legal suffixes removed — used for matching)
  * clean title-case version (used later in chunks)

This is where the “extra” company columns come from — and why they can safely be dropped later.

**Category standardisation.**
Explicit mapping dicts for all categoricals (`status_map`, `lead_source_map`, `priority_map`, etc.).
Raw data had variants like `"ad"` vs `"ads"`, `"web"` vs `"website"`, `"in_progress"` vs `"in progress"`. Without this, identical concepts produce different embeddings.

**Deduplication.**

* Exact duplicate rows dropped
* Then per-ID deduplication keeping the most complete or most recent record
* For tickets: prioritises rows with more non-null fields over newer but sparse ones

**Quality flags and derived metrics.**
Computed once, from clean data:

* `has_valid_email`, `is_resolved`, `is_low_information_ticket`
* word counts
* `days_since_meeting`
* `ticket_risk_score`

These get carried forward into the next stage.

**First-pass search text.**
Each dataset gets a basic pipe-delimited text column (`customer_search_text`, `support_ticket_chunk`, etc.).
This is a placeholder — not good enough for RAG, but useful as an intermediate.

---

### Stage 2 — structured → retrieval-ready (improvement scripts)

This is where the data becomes usable for RAG.

The main idea here is simple: **transform clean data into readable, retrieval-ready chunks with strong signal and filterable structure.**

For each dataset:

**Drop zero-variance and redundant columns.**
Stage 1 produced useful flags — but some ended up being all-True or all-False across the dataset. These add nothing to embeddings, so they’re removed.
Same for duplicate company fields — keep one clean version, drop the rest. Less noise = better signal.

**Rebuild the text column from scratch.**
All first-pass search columns are dropped and replaced with a single structured `rag_chunk`.

Each chunk is formatted as labeled key-value lines. This is intentional:

* Embedding models handle `"Field: Value"` better than unstructured prose
* Still human-readable if surfaced in a UI

**Attach a `metadata` JSON column to every row.**

* `rag_chunk` → embedded
* `metadata` → stored alongside vector

This separation enables **filtered retrieval**:
Instead of “hope semantic search finds the right thing,” you can pre-filter by:
`sales_rep`, `country`, `priority`, `status`, etc.

**Flag data quality issues instead of dropping rows.**
I found repeated template text across notes and tickets — identical content across many rows.

If embedded as-is:
→ multiple records collapse to near-identical vectors
→ retrieval becomes arbitrary

Instead of deleting them:

* Flag `is_recycled_note`, `is_recycled_subject`, `is_recycled_description`
* Add warnings inside the chunk (⚠️)
* Store flags in metadata for filtering

Same idea for unresolved tickets:

* `has_real_resolution = False`
  → lets retrieval exclude them when searching for solutions

**Add ordinal representations for ranked categories.**
Fields like:

* `priority` (Low → Critical)
* `budget_range` (Below 5K → 50K+)

These are ordered, but stored as strings.

Vector DBs can’t range-filter strings, so:

* Map to integers (`priority_rank`, `budget_rank`)
* Store in metadata

Now queries like:
“High or Critical tickets for rep X”
can be handled as pre-filters instead of post-processing.

---

### Dataset-specific enrichment

Where the data justified it:

**Sales notes**
Regex-based topic extraction (CRM, Pricing, Demo, Budget, etc.)
Stored both in:

* chunk (explicit signal)
* metadata (filterable list)

Free text alone doesn’t guarantee retrieval surfaces these terms — making them explicit does.

**Customers**

* Normalised inconsistent categoricals (country, lead source casing)
* Added `company_size_bucket` (Small / Mid-size / Large)

A raw number (employee count) has weak semantic meaning. A label gives context.

**Support tickets**
Two chunk variants per row:

* `rag_chunk_full` → includes resolution (for solved-case lookup)
* `rag_chunk_issue` → excludes resolution (for similarity on open issues)

Using one chunk would contaminate similarity search with resolution text.

---

### Why two stages?

Deliberate separation of concerns.

* **Stage 1 is reusable** — clean, structured data that works for analytics, reporting, or any downstream system
* **Stage 2 is RAG-specific** — focused entirely on retrieval quality

If the retrieval strategy changes, you rerun stage 2.
You don’t touch stage 1.







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
