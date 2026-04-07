# Local RAG System  
**Retrieval-Augmented Generation over Structured Business Data**

**Tech Stack:** Python 3.11 · ChromaDB · Ollama (LLaMA 3) · SentenceTransformers · nomic-embed-text-v1  

---

## Overview

This project implements a fully local Retrieval-Augmented Generation (RAG) system designed for querying structured business data such as CRM records, support tickets, sales notes, and email threads.

The system is built around a core principle:

> **RAG quality is determined far more by data design and retrieval strategy than by model choice.**

Instead of treating RAG as a simple “embed + search” problem, this system focuses on:

- structured data transformation  
- metadata-aware retrieval  
- entity-safe querying  
- deterministic filtering and ranking  

---

## System Architecture

The system is split into two independent pipelines:

### 1. Ingestion Pipeline (`ingest.py`)
Transforms raw data into structured, high-signal chunks and stores embeddings in ChromaDB.

### 2. Query Pipeline (`query.py`)
Handles user queries, retrieves relevant context, and generates grounded answers using LLaMA 3 via Ollama.

---

## End-to-End Query Flow
User Query
↓
Query Parsing (entity + type extraction)
↓
Embedding (search_query prefix)
↓
Vector Search (ChromaDB)
↓
Metadata Filtering (type + customer scope)
↓
Deduplication + Re-ranking
↓
Context Construction
↓
LLM (LLaMA 3 via Ollama)
↓
Answer + Sources

---

## Key Design Decisions

### 1. Entity-Aware Retrieval (Customer Isolation)

Queries automatically detect and normalize customer IDs (e.g. `customer 27 → 027`) and enforce strict scoping:

- Filters applied at query time (`customer_id_norm`)
- Only “single_customer” chunks are allowed  
- Guardrails prevent cross-customer data mixing  

**Why it matters:**  
Prevents one of the most common RAG failures: mixing data across entities.

---

### 2. Metadata Filtering Before Similarity Search

Structured filters (record type, customer, etc.) are applied directly during vector search.

**Why it matters:**

- Reduces search space  
- Improves precision vs pure vector similarity  
- Makes retrieval deterministic and debuggable  

---

### 3. Structured Chunking (Not Free Text)

All data is transformed into structured, labeled chunks:

- **CSV:** schema, preview, row, row_group, column_stats  
- **JSON:** thread summaries, individual messages, flattened paths  
- **TXT:** paragraph windows with overlap  

Example chunk types:

- `row` → atomic records  
- `row_group` → precomputed aggregations  
- `thread_summary` → email-level context  
- `individual_message` → fine-grained detail  

**Why it matters:**

- Improves embedding consistency  
- Enables reasoning over structured data  
- Makes outputs interpretable and auditable  

---

### 4. Customer-Safe Aggregations

Instead of scanning full datasets at query time:

- Rows are grouped per customer during ingestion  
- Aggregations are stored as `row_group` chunks  
- Groups never mix customers  

**Why it matters:**

- Enables scalable aggregation queries  
- Prevents incomplete answers  
- Keeps latency low  

---

### 5. Deduplication + Heuristic Re-ranking

Retrieved chunks are:

- deduplicated using content fingerprints  
- re-ranked using metadata-aware scoring  

Scoring priorities:

- +100 → exact customer match  
- +20 → row-level data  
- +15 → messages  
- +10 → grouped data  

**Why it matters:**

- Reduces noise  
- Prioritizes high-signal context  
- Stabilizes outputs  

---

### 6. Embedding Strategy

- Model: `nomic-embed-text-v1`  
- Prefixing strategy:
  - Documents → `search_document:`  
  - Queries → `search_query:`  

**Why it matters:**  
Aligns query and document embeddings in the same semantic space.

---

### 7. Robust Multi-Format Ingestion

Supports:

- **CSV**
  - schema + preview + row + grouped + stats chunks  
- **JSON**
  - email threads (structured + temporal)  
  - generic JSON (flattened paths)  
- **TXT**
  - paragraph-aware chunking with overlap  

Includes:

- safe parsing (multiple CSV strategies)  
- normalization (IDs, whitespace, metadata)  
- validation and chunk integrity checks  

---

## Prompting Strategy

The LLM is constrained to:

- answer **only using retrieved context**  
- explicitly state when data is missing  
- avoid guessing or hallucination  
- enforce strict entity boundaries  

---

## Example

**Query**
[ticket] How many open tickets does customer 027 have?

**Output**
Customer 027 has 12 open tickets.

Sources:

data/tickets/ticket_123.csv
data/tickets/ticket_456.csv


---

## Limitations

- No hybrid search (BM25 + vector)  
- No cross-encoder re-ranking  
- Aggregations limited by retrieved chunks  
- No conversation memory  
- Performance limited by local LLM  
- No PDF ingestion  

---

## Future Improvements

- Hybrid retrieval (BM25 + vector)  
- Cross-encoder re-ranking  
- Query decomposition for complex queries  
- Streaming responses  
- Automated re-ingestion pipelines  
- PDF and unstructured document support  


