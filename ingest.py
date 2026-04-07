import os
import re
import json
import math
import hashlib
from pathlib import Path
from typing import Any, Iterable

import torch
import chromadb
import pandas as pd
from pandas.errors import ParserError
from sentence_transformers import SentenceTransformer

# Limit CPU threads before heavy imports
torch.set_num_threads(2)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Config ---
DATA_FOLDER = "data"
CHROMA_PATH = "chroma_store"
COLLECTION_NAME = "my_docs"

EMBED_MODEL_NAME = "nomic-ai/nomic-embed-text-v1"

EMBED_BATCH_SIZE = 8
CHROMA_UPSERT_BATCH = 100

# Chunking
TXT_TARGET_CHARS = 1200
TXT_OVERLAP_CHARS = 200
JSON_MAX_PATHS_PER_CHUNK = 80
CSV_ROW_GROUP_SIZE = 20

# Limits
MAX_CELL_CHARS = 500
MAX_TEXT_CHARS = 6000

# Columns used for per-column stats summary chunks
CSV_STATS_COLUMNS = [
    "status", "priority", "issue_type", "country",
    "industry", "assigned_to", "assigned_sales_rep", "sales_rep",
    "plan_interest", "interest_area",
]


# ---------------- Utility helpers ----------------

def rel_source_path(filepath: str) -> str:
    return os.path.relpath(filepath, DATA_FOLDER)


def stable_chunk_id(filepath: str, chunk_index: int, text: str) -> str:
    rel_path = rel_source_path(filepath)
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:10]
    return f"{rel_path}::chunk::{chunk_index}::{digest}"


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    if len(text) > MAX_CELL_CHARS:
        text = text[:MAX_CELL_CHARS] + " ..."
    return text


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def detect_type(path: str) -> str:
    parts = {p.lower() for p in Path(path).parts}
    if "proposals" in parts:
        return "proposal"
    if "policies" in parts:
        return "policy"
    if "faqs" in parts:
        return "faq"
    if "services" in parts:
        return "service_knowledge"
    return "document"


def normalize_customer_id(value: Any) -> str:
    """
    Normalize customer IDs so 27, 027, cust_027, customer 027 all become 027.
    If no digits exist, returns stripped string as-is.
    """
    text = safe_str(value)
    if not text:
        return ""
    m = re.search(r"(\d+)", text)
    if m:
        return m.group(1).zfill(3)
    return text.lower()


def normalize_metadata(metadata: dict) -> dict:
    normalized = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            normalized[key] = value
        else:
            normalized[key] = str(value)
    return normalized


def validate_chunk(chunk: dict, source_path: str) -> bool:
    if not isinstance(chunk, dict):
        print(f"  Invalid chunk from {source_path}: chunk is not a dict")
        return False
    if "text" not in chunk or "metadata" not in chunk:
        print(f"  Invalid chunk from {source_path}: missing 'text' or 'metadata'")
        return False
    if not isinstance(chunk["text"], str):
        print(f"  Invalid chunk from {source_path}: 'text' must be a string")
        return False
    if not isinstance(chunk["metadata"], dict):
        print(f"  Invalid chunk from {source_path}: 'metadata' must be a dict")
        return False
    if not chunk["text"].strip():
        print(f"  Invalid chunk from {source_path}: empty text")
        return False
    return True


def make_base_metadata(path: str, source_format: str, record_type: str) -> dict:
    return {
        "source_path": rel_source_path(path),
        "source_file": os.path.basename(path),
        "source_format": source_format,
        "record_type": record_type,
    }


def chunk_text_with_overlap(text: str, target_chars: int, overlap_chars: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= target_chars:
        return [text]

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + target_chars, n)
        if end < n:
            split_candidates = [
                text.rfind("\n\n", start, end),
                text.rfind("\n", start, end),
                text.rfind(". ", start, end),
                text.rfind(" ", start, end),
            ]
            best = max(split_candidates)
            if best > start + target_chars // 2:
                end = best + (2 if text[best:best + 2] == ". " else 1)

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= n:
            break

        start = max(end - overlap_chars, start + 1)

    return chunks


def batched(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def ingest_jsonl(filepath: str):
    print(f"\nLoading preprocessed chunks from {filepath}")
    chunks = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            if not validate_chunk(chunk, filepath):
                continue
            chunks.append(chunk)

    print(f"Loaded {len(chunks)} chunks from JSONL")
    return chunks


# ---------------- JSON ingestion ----------------

def is_email_thread_like(obj: dict) -> bool:
    return isinstance(obj, dict) and isinstance(obj.get("messages"), list)


def flatten_json(
    obj: Any,
    prefix: str = "",
    out: list[tuple[str, str]] | None = None,
) -> list[tuple[str, str]]:
    if out is None:
        out = []

    if isinstance(obj, dict):
        if not obj:
            out.append((prefix or "$", "{}"))
        for key, value in obj.items():
            key = str(key)
            next_prefix = f"{prefix}.{key}" if prefix else key
            flatten_json(value, next_prefix, out)
    elif isinstance(obj, list):
        if not obj:
            out.append((prefix or "$", "[]"))
        for i, value in enumerate(obj):
            next_prefix = f"{prefix}[{i}]" if prefix else f"[{i}]"
            flatten_json(value, next_prefix, out)
    else:
        out.append((prefix or "$", safe_str(obj)))

    return out


def load_json_email_thread(path: str, item: dict, item_index: int) -> list[dict]:
    documents = []

    messages = item.get("messages", [])
    valid_messages = [m for m in messages if isinstance(m, dict)]
    valid_messages = sorted(valid_messages, key=lambda m: str(m.get("timestamp", "")))

    customer_id = safe_str(item.get("customer_id"))
    customer_id_norm = normalize_customer_id(customer_id)

    thread_metadata = {
        **make_base_metadata(path, "json", "email_thread"),
        "json_mode": "email_thread",
        "item_index": item_index,
        "thread_id": safe_str(item.get("thread_id")),
        "customer_id": customer_id,
        "customer_id_norm": customer_id_norm,
        "company_name": safe_str(item.get("company_name")),
        "subject": safe_str(item.get("subject")),
        "message_count": len(valid_messages),
        "entity_scope": "single_customer" if customer_id_norm else "unknown",
    }

    thread_summary = {k: v for k, v in item.items() if k != "messages"}
    thread_summary["message_count"] = len(valid_messages)

    if valid_messages:
        thread_summary["first_message_time"] = valid_messages[0].get("timestamp", "")
        thread_summary["last_message_time"] = valid_messages[-1].get("timestamp", "")
        participants = sorted({
            safe_str(m.get("from_email"))
            for m in valid_messages
            if safe_str(m.get("from_email"))
        })
        thread_summary["participants"] = participants

    summary_lines = [
        f"File: {os.path.basename(path)}",
        "JSON object type: email_thread",
        f"Item index: {item_index}",
        f"Customer ID: {customer_id_norm or customer_id}",
        "Thread summary:",
    ]
    for k, v in thread_summary.items():
        summary_lines.append(f"{k}: {safe_str(v)}")

    documents.append({
        "text": "\n".join(summary_lines)[:MAX_TEXT_CHARS],
        "metadata": {**thread_metadata, "chunk_type": "thread_summary"},
    })

    for i, msg in enumerate(valid_messages):
        msg_text = (
            f"File: {os.path.basename(path)}\n"
            f"JSON object type: email_thread\n"
            f"Item index: {item_index}\n"
            f"Message index: {i}\n"
            f"Customer ID: {customer_id_norm or customer_id}\n"
            f"Thread ID: {safe_str(item.get('thread_id'))}\n"
            f"Subject: {safe_str(item.get('subject'))}\n"
            f"Company: {safe_str(item.get('company_name'))}\n"
            f"From: {safe_str(msg.get('from_name'))} <{safe_str(msg.get('from_email'))}>\n"
            f"To: {safe_str(msg.get('to_name'))} <{safe_str(msg.get('to_email'))}>\n"
            f"Date: {safe_str(msg.get('timestamp'))}\n\n"
            f"{safe_str(msg.get('body'))}"
        )
        documents.append({
            "text": msg_text[:MAX_TEXT_CHARS],
            "metadata": {
                **thread_metadata,
                "chunk_type": "individual_message",
                "message_index": i,
                "from_email": safe_str(msg.get("from_email")),
                "from_name": safe_str(msg.get("from_name")),
                "timestamp": safe_str(msg.get("timestamp")),
            },
        })

    flat_pairs = flatten_json(item)
    path_lines = [
        f"File: {os.path.basename(path)}",
        "JSON object type: email_thread",
        f"Item index: {item_index}",
        f"Customer ID: {customer_id_norm or customer_id}",
        "Flattened paths:",
    ] + [f"{k} = {v}" for k, v in flat_pairs]

    header = "\n".join(path_lines[:5])
    body_lines = path_lines[5:]

    for chunk_i, start in enumerate(range(0, len(body_lines), JSON_MAX_PATHS_PER_CHUNK)):
        chunk_lines = [header] + body_lines[start:start + JSON_MAX_PATHS_PER_CHUNK]
        documents.append({
            "text": "\n".join(chunk_lines)[:MAX_TEXT_CHARS],
            "metadata": {
                **thread_metadata,
                "chunk_type": "flattened_paths",
                "path_chunk_index": chunk_i,
            },
        })

    return documents


def load_json_generic(path: str, item: Any, item_index: int) -> list[dict]:
    documents = []

    record_type = "json_object" if isinstance(item, dict) else "json_value"
    base_meta = {
        **make_base_metadata(path, "json", record_type),
        "json_mode": "generic",
        "item_index": item_index,
        "entity_scope": "unknown",
    }

    flat_pairs = flatten_json(item)

    top_keys = []
    if isinstance(item, dict):
        top_keys = list(item.keys())
    elif isinstance(item, list):
        top_keys = [f"[{i}]" for i in range(min(len(item), 20))]

    summary_lines = [
        f"File: {os.path.basename(path)}",
        "JSON object type: generic",
        f"Item index: {item_index}",
        f"Top-level kind: {type(item).__name__}",
        f"Top-level keys: {', '.join(map(str, top_keys[:50]))}" if top_keys else "Top-level keys: none",
        f"Flattened path count: {len(flat_pairs)}",
    ]
    documents.append({
        "text": "\n".join(summary_lines),
        "metadata": {**base_meta, "chunk_type": "summary"},
    })

    path_lines = [f"{path_} = {value}" for path_, value in flat_pairs]
    for chunk_i, start in enumerate(range(0, len(path_lines), JSON_MAX_PATHS_PER_CHUNK)):
        chunk_lines = [
            f"File: {os.path.basename(path)}",
            "JSON object type: generic",
            f"Item index: {item_index}",
            "Flattened paths:",
            *path_lines[start:start + JSON_MAX_PATHS_PER_CHUNK],
        ]
        documents.append({
            "text": "\n".join(chunk_lines)[:MAX_TEXT_CHARS],
            "metadata": {**base_meta, "chunk_type": "flattened_paths", "path_chunk_index": chunk_i},
        })

    return documents


def load_json(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data if isinstance(data, list) else [data]
    documents = []

    for item_index, item in enumerate(items):
        if is_email_thread_like(item):
            documents.extend(load_json_email_thread(path, item, item_index))
        else:
            documents.extend(load_json_generic(path, item, item_index))

    return documents


# ---------------- CSV ingestion ----------------

def infer_csv_record_type(path: str) -> tuple[str, str | None]:
    file_name = os.path.basename(path)
    folder_name = os.path.basename(os.path.dirname(path)).lower()
    lower_name = file_name.lower()

    if folder_name == "tickets":
        return "support_ticket", "ticket_id"
    if folder_name == "sales_notes":
        return "sales_note", "note_id"
    if folder_name == "crm_records":
        if "customer" in lower_name:
            return "customer", "customer_id"
        if "lead" in lower_name:
            return "lead", "lead_id"
        return "crm_record", None

    return "record", None


def read_csv_safely(path: str) -> pd.DataFrame:
    attempts = [
        {"sep": None, "engine": "python", "encoding": "utf-8", "on_bad_lines": "skip"},
        {"sep": ",", "engine": "python", "encoding": "utf-8", "on_bad_lines": "skip"},
        {"sep": ";", "engine": "python", "encoding": "utf-8", "on_bad_lines": "skip"},
        {"sep": None, "engine": "python", "encoding": "latin-1", "on_bad_lines": "skip"},
    ]
    last_error = None
    for kwargs in attempts:
        try:
            return pd.read_csv(path, **kwargs)
        except Exception as e:
            last_error = e
    raise ValueError(f"Could not read CSV '{path}': {last_error}")


def stringify_row(row: dict[str, Any], columns: list[str]) -> str:
    parts = []
    for col in columns:
        value = safe_str(row.get(col))
        if value:
            parts.append(f"{col}: {value}")
    return " | ".join(parts)


def load_csv(path: str) -> list[dict]:
    chunks = []
    df = read_csv_safely(path)
    file_name = os.path.basename(path)
    folder_name = os.path.basename(os.path.dirname(path)).lower()
    record_type, id_field = infer_csv_record_type(path)

    df.columns = [safe_str(c) for c in df.columns]
    columns = list(df.columns)

    # Normalize customer_id if present
    if "customer_id" in df.columns:
        df["customer_id"] = df["customer_id"].apply(safe_str)
        df["customer_id_norm"] = df["customer_id"].apply(normalize_customer_id)

    base_meta = {
        **make_base_metadata(path, "csv", record_type),
        "folder": folder_name,
        "row_count": int(len(df)),
        "column_count": int(len(columns)),
    }

    # Schema chunk
    schema_lines = [
        f"File: {file_name}",
        "Format: csv",
        f"Folder: {folder_name}",
        f"Record type: {record_type}",
        f"Rows: {len(df)}",
        f"Columns: {len(columns)}",
        "Column names:",
        *[f"- {col}" for col in columns],
    ]
    chunks.append({
        "text": "\n".join(schema_lines)[:MAX_TEXT_CHARS],
        "metadata": {**base_meta, "chunk_type": "schema", "entity_scope": "mixed"},
    })

    # Preview chunk
    preview_rows = df.head(min(5, len(df))).fillna("").to_dict(orient="records")
    preview_lines = [f"File: {file_name}", "Format: csv", "Preview rows:"]
    for i, row in enumerate(preview_rows):
        preview_lines.append(f"Row {i}: {stringify_row(row, columns + (['customer_id_norm'] if 'customer_id_norm' in df.columns else []))}")
    chunks.append({
        "text": "\n".join(preview_lines)[:MAX_TEXT_CHARS],
        "metadata": {**base_meta, "chunk_type": "preview", "entity_scope": "mixed"},
    })

    # Per-column stats chunks
    for col in CSV_STATS_COLUMNS:
        if col not in df.columns:
            continue
        counts = df[col].dropna().astype(str).str.strip()
        counts = counts[counts != ""].value_counts().to_dict()
        if not counts:
            continue
        lines = [
            f"File: {file_name}",
            "Format: csv",
            f"Record type: {record_type}",
            f"Total rows: {len(df)}",
            f"Column value counts — {col}:",
        ] + [f"  {k}: {v}" for k, v in sorted(counts.items())]
        chunks.append({
            "text": "\n".join(lines)[:MAX_TEXT_CHARS],
            "metadata": {**base_meta, "chunk_type": "column_stats", "stats_column": col, "entity_scope": "mixed"},
        })

    # Row chunks
    records = df.fillna("").to_dict(orient="records")
    for row_index, row in enumerate(records):
        row_text = stringify_row(row, list(row.keys()))
        if not row_text:
            continue

        record_id = "unknown"
        if id_field and safe_str(row.get(id_field)):
            record_id = safe_str(row.get(id_field))
        else:
            for fallback in ["ticket_id", "customer_id", "lead_id", "note_id", "id"]:
                if safe_str(row.get(fallback)):
                    record_id = safe_str(row.get(fallback))
                    break

        customer_id = safe_str(row.get("customer_id"))
        customer_id_norm = normalize_customer_id(customer_id)

        chunk_text = (
            f"File: {file_name}\n"
            f"Format: csv\n"
            f"Record type: {record_type}\n"
            f"Row index: {row_index}\n"
            f"Customer ID: {customer_id_norm or customer_id}\n"
            f"Columns: {', '.join(row.keys())}\n"
            f"Row data: {row_text}"
        )
        metadata = {
            **base_meta,
            "chunk_type": "row",
            "row_index": row_index,
            "id": record_id,
            "entity_scope": "single_customer" if customer_id_norm else "unknown",
        }

        for field in [
            "customer_id", "company_name", "status", "priority", "issue_type",
            "industry", "country", "assigned_to", "assigned_sales_rep", "sales_rep",
            "plan_interest", "interest_area", "meeting_date", "created_at",
        ]:
            value = safe_str(row.get(field))
            if value:
                metadata[field] = value

        if customer_id_norm:
            metadata["customer_id_norm"] = customer_id_norm

        chunks.append({"text": chunk_text[:MAX_TEXT_CHARS], "metadata": metadata})

    # Customer-safe row-group chunks:
    # For datasets with customer_id, group WITHIN a single customer only.
    if "customer_id_norm" in df.columns:
        grouped = df.fillna("").groupby("customer_id_norm", dropna=False, sort=False)
        for customer_id_norm, group_df in grouped:
            if not safe_str(customer_id_norm):
                continue

            customer_records = group_df.to_dict(orient="records")
            for group_start in range(0, len(customer_records), CSV_ROW_GROUP_SIZE):
                group = customer_records[group_start:group_start + CSV_ROW_GROUP_SIZE]
                row_indexes = [int(r.get("_row_index")) for r in group if "_row_index" in r]

                group_lines = [
                    f"File: {file_name}",
                    "Format: csv",
                    f"Record type: {record_type}",
                    f"Customer ID: {customer_id_norm}",
                    f"Grouped rows for one customer only",
                    f"Columns: {', '.join(group[0].keys()) if group else ', '.join(columns)}",
                ]
                for row in group:
                    row_idx = row.get("_row_index", "?")
                    row_text = stringify_row(row, list(row.keys()))
                    if row_text:
                        group_lines.append(f"Row {row_idx}: {row_text}")

                chunks.append({
                    "text": "\n".join(group_lines)[:MAX_TEXT_CHARS],
                    "metadata": {
                        **base_meta,
                        "chunk_type": "row_group",
                        "customer_id_norm": customer_id_norm,
                        "entity_scope": "single_customer",
                        "row_group_start": row_indexes[0] if row_indexes else group_start,
                        "row_group_end": row_indexes[-1] if row_indexes else group_start + len(group) - 1,
                    },
                })
    else:
        # Fallback for datasets without customer_id
        for group_start in range(0, len(records), CSV_ROW_GROUP_SIZE):
            group = records[group_start:group_start + CSV_ROW_GROUP_SIZE]
            group_lines = [
                f"File: {file_name}",
                "Format: csv",
                f"Record type: {record_type}",
                f"Row group: {group_start} to {group_start + len(group) - 1}",
                f"Columns: {', '.join(columns)}",
            ]
            for i, row in enumerate(group, start=group_start):
                row_text = stringify_row(row, columns)
                if row_text:
                    group_lines.append(f"Row {i}: {row_text}")
            chunks.append({
                "text": "\n".join(group_lines)[:MAX_TEXT_CHARS],
                "metadata": {
                    **base_meta,
                    "chunk_type": "row_group",
                    "row_group_start": group_start,
                    "row_group_end": group_start + len(group) - 1,
                    "entity_scope": "mixed",
                },
            })

    return chunks


# ---------------- TXT ingestion ----------------

def load_txt(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    file_type = detect_type(path)
    file_name = os.path.basename(path)
    rel_path = rel_source_path(path)
    base_meta = make_base_metadata(path, "txt", file_type)

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    merged = "\n\n".join(paragraphs) if paragraphs else text.strip()

    text_chunks = chunk_text_with_overlap(
        merged,
        target_chars=TXT_TARGET_CHARS,
        overlap_chars=TXT_OVERLAP_CHARS,
    )

    documents = []
    summary_lines = [
        f"File: {file_name}",
        f"Path: {rel_path}",
        "Format: txt",
        f"Record type: {file_type}",
        f"Character count: {len(text)}",
        f"Chunk count: {len(text_chunks)}",
    ]
    documents.append({
        "text": "\n".join(summary_lines),
        "metadata": {**base_meta, "chunk_type": "summary", "entity_scope": "unknown"},
    })

    for i, chunk_text in enumerate(text_chunks):
        doc_text = (
            f"File: {file_name}\n"
            f"Path: {rel_path}\n"
            f"Format: txt\n"
            f"Record type: {file_type}\n"
            f"Chunk index: {i}\n\n{chunk_text}"
        )
        documents.append({
            "text": doc_text[:MAX_TEXT_CHARS],
            "metadata": {**base_meta, "chunk_type": "paragraph_window", "chunk_index": i, "entity_scope": "unknown"},
        })

    return documents


# ---------------- File dispatch ----------------

def load_file(path: str) -> list[dict]:
    ext = os.path.splitext(path)[1].lower()
    loaders = {".csv": load_csv, ".json": load_json, ".txt": load_txt}
    if ext not in loaders:
        raise ValueError(f"Unsupported file type: {ext}")
    return loaders[ext](path)


# ---------------- Embedding helpers ----------------

def embed_texts(embedder: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    prefixed = [f"search_document: {t}" for t in texts]

    all_embeddings = []
    total = len(prefixed)
    print(f"Embedding {total} chunks in batches of {EMBED_BATCH_SIZE}...")

    for i, batch in enumerate(batched(prefixed, EMBED_BATCH_SIZE)):
        embeddings = embedder.encode(
            batch,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        all_embeddings.extend(embeddings.tolist())
        done = min((i + 1) * EMBED_BATCH_SIZE, total)
        print(f"  [{done}/{total}] chunks embedded", end="\r")

    print()
    return all_embeddings


# ---------------- Skip-already-ingested logic ----------------

def get_existing_ids(collection) -> set[str]:
    result = collection.get(include=[])
    return set(result["ids"])


def filter_new_chunks(
    all_chunks: list[str],
    all_ids: list[str],
    all_metadata: list[dict],
    existing_ids: set[str],
) -> tuple[list[str], list[str], list[dict]]:
    new_chunks, new_ids, new_metadata = [], [], []
    for chunk, id_, meta in zip(all_chunks, all_ids, all_metadata):
        if id_ not in existing_ids:
            new_chunks.append(chunk)
            new_ids.append(id_)
            new_metadata.append(meta)
    return new_chunks, new_ids, new_metadata


# ---------------- Main ingestion ----------------

def ingest_documents() -> None:
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
        print(f"Created '{DATA_FOLDER}/' folder. Add your files there and re-run.")
        return

    all_chunks: list[str] = []
    all_ids: list[str] = []
    all_metadata: list[dict] = []

    found_files = False

    jsonl_path = os.path.join(DATA_FOLDER, "rag_chunks.jsonl")

    if os.path.exists(jsonl_path):
        found_files = True
        print(f"\nLoading preprocessed chunks from '{jsonl_path}'...")

        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                loaded_count = 0

                for line in f:
                    chunk = json.loads(line)

                    if validate_chunk(chunk, jsonl_path):
                        chunk["metadata"] = normalize_metadata(chunk["metadata"])

                        # Backfill normalized customer ID if applicable
                        if "customer_id" in chunk["metadata"] and "customer_id_norm" not in chunk["metadata"]:
                            chunk["metadata"]["customer_id_norm"] = normalize_customer_id(chunk["metadata"]["customer_id"])

                        all_chunks.append(chunk["text"])
                        all_ids.append(chunk["id"])
                        all_metadata.append(chunk["metadata"])
                        loaded_count += 1

                print(f"  Loaded {loaded_count} chunks from rag_chunks.jsonl")

        except Exception as e:
            print(f"  Failed loading rag_chunks.jsonl: {e}")

    print(f"\nScanning '{DATA_FOLDER}/'...")
    for root, _, files in os.walk(DATA_FOLDER):
        for filename in files:
            filepath = os.path.join(root, filename)

            if filepath == jsonl_path:
                continue

            found_files = True

            try:
                chunks = load_file(filepath)
            except ValueError as e:
                print(f"  Skipped {rel_source_path(filepath)}: {e}")
                continue
            except ParserError as e:
                print(f"  Failed {rel_source_path(filepath)}: parser error: {e}")
                continue
            except json.JSONDecodeError as e:
                print(f"  Failed {rel_source_path(filepath)}: invalid JSON: {e}")
                continue
            except Exception as e:
                print(f"  Failed {rel_source_path(filepath)}: {e}")
                continue

            valid_chunks = []
            for chunk in chunks:
                if validate_chunk(chunk, filepath):
                    chunk["metadata"] = normalize_metadata(chunk["metadata"])
                    valid_chunks.append(chunk)

            if not valid_chunks:
                print(f"  {rel_source_path(filepath)}: no valid chunks produced")
                continue

            chunk_type = valid_chunks[0]["metadata"].get("record_type", "unknown")
            print(f"  {rel_source_path(filepath)}: {len(valid_chunks)} chunks (type: {chunk_type})")

            for i, chunk in enumerate(valid_chunks):
                all_chunks.append(chunk["text"])
                all_ids.append(stable_chunk_id(filepath, i, chunk["text"]))
                all_metadata.append(chunk["metadata"])

    if not found_files:
        print(f"No files found in '{DATA_FOLDER}/'. Add some and re-run.")
        return

    if not all_chunks:
        print("No valid chunks found. Nothing to ingest.")
        return

    print(f"\nConnecting to Chroma at '{CHROMA_PATH}'...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    existing_ids = get_existing_ids(collection)
    print(f"Already stored: {len(existing_ids)} chunks")

    all_chunks, all_ids, all_metadata = filter_new_chunks(
        all_chunks, all_ids, all_metadata, existing_ids
    )

    if not all_chunks:
        print("Nothing new to ingest — all chunks already in Chroma.")
        return

    print(f"New chunks to embed and store: {len(all_chunks)}")

    print("\nLoading embedding model...")
    embedder = SentenceTransformer(
        EMBED_MODEL_NAME,
        trust_remote_code=True,
        device="cpu",
    )

    all_embeddings = embed_texts(embedder, all_chunks)

    print(f"\nUpserting {len(all_chunks)} chunks into Chroma...")
    for i in range(0, len(all_chunks), CHROMA_UPSERT_BATCH):
        collection.upsert(
            documents=all_chunks[i:i + CHROMA_UPSERT_BATCH],
            embeddings=all_embeddings[i:i + CHROMA_UPSERT_BATCH],
            ids=all_ids[i:i + CHROMA_UPSERT_BATCH],
            metadatas=all_metadata[i:i + CHROMA_UPSERT_BATCH],
        )
        done = min(i + CHROMA_UPSERT_BATCH, len(all_chunks))
        print(f"  [{done}/{len(all_chunks)}] upserted", end="\r")

    print()
    print(f"\nDone! {len(all_chunks)} new chunks stored in Chroma collection '{COLLECTION_NAME}'.")
    print(f"Total in collection: {len(existing_ids) + len(all_chunks)}")


if __name__ == "__main__":
    ingest_documents()
