"""
RAG Dataset Improvement Script for sales_notes_cleaned.csv
-----------------------------------------------------------
Issues fixed:
  1. Removes redundant columns (note_summary_short == note_text, triple company name columns)
  2. Drops the near-duplicate sales_note_text (keeping sales_note_chunk as base)
  3. Enriches each chunk with structured context to give embeddings more signal
  4. Adds a `rag_chunk` column — the single column to embed and index
  5. Adds a `metadata` column (JSON) — to attach to each vector for filtered retrieval
  6. Flags short notes that may be too thin for reliable retrieval
"""

import pandas as pd
import json
import re

INPUT_PATH = "sales_notes_cleaned.csv"
OUTPUT_PATH = "sales_notes_rag_ready.csv"

# ── 1. Load ────────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_PATH)
print(f"Loaded {len(df)} rows, {df.shape[1]} columns")

# ── 2. Drop redundant / noisy columns ─────────────────────────────────────────
cols_to_drop = [
    "note_summary_short",       # 100% identical to note_text
    "company_name_raw",         # superseded by company_name_normalized
    "sales_note_text",          # near-duplicate of sales_note_chunk
    "sales_note_chunk",         # will be rebuilt below, cleaner
    "meeting_month",            # derivable from meeting_date; adds nothing for RAG
    "has_customer_id",          # all True — zero variance
    "has_note_text",            # all True — zero variance
    "has_meeting_date",         # all True — zero variance
    "is_low_information_note",  # all False — zero variance
]
df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
print(f"After dropping redundant columns: {df.shape[1]} columns remain")

# ── 3. Parse meeting_date cleanly ─────────────────────────────────────────────
df["meeting_date_parsed"] = pd.to_datetime(df["meeting_date"], utc=True)
df["meeting_date_str"] = df["meeting_date_parsed"].dt.strftime("%B %d, %Y")  # e.g. "May 11, 2025"

# ── 4. Extract topics mentioned in note_text ──────────────────────────────────
TOPIC_KEYWORDS = {
    "CRM": r"\bCRM\b",
    "Pricing": r"\bpric(e|ing)\b",
    "Demo": r"\bdemo\b",
    "Security": r"\bsecurity\b",
    "Implementation": r"\bimplementation\b",
    "Integration": r"\bintegrat\w+\b",
    "Timeline": r"\btimeline\b",
    "Pilot": r"\bpilot\b",
    "Budget": r"\bbudget\b",
    "Multi-branch": r"\bmulti.branch\b",
    "Training": r"\btraining\b",
    "Follow-up": r"\bfollow.up\b",
}

def extract_topics(text: str) -> str:
    found = [topic for topic, pattern in TOPIC_KEYWORDS.items()
             if re.search(pattern, text, re.IGNORECASE)]
    return ", ".join(found) if found else "General"

df["topics"] = df["note_text"].apply(extract_topics)

# ── 5. Build a richer RAG chunk ────────────────────────────────────────────────
# Goal: give the embedding model more semantic signal per note, while keeping
# the format human-readable (useful if you display retrieved chunks in a UI).

def build_rag_chunk(row: pd.Series) -> str:
    lines = [
        f"Sales Note | {row['note_id']} | {row['company_name']} | {row['sales_rep']} | {row['meeting_date_str']}",
        f"Company (normalized): {row['company_name_normalized']}",
        f"Topics discussed: {row['topics']}",
        f"Note: {row['note_text']}",
    ]
    return "\n".join(lines)

df["rag_chunk"] = df.apply(build_rag_chunk, axis=1)

# ── 6. Build a metadata dict per row (for vector DB filtered retrieval) ────────
def build_metadata(row: pd.Series) -> str:
    meta = {
        "note_id":           row["note_id"],
        "customer_id":       row["customer_id"],
        "company_name":      row["company_name"],
        "company_normalized":row["company_name_normalized"],
        "sales_rep":         row["sales_rep"],
        "meeting_date":      row["meeting_date_str"],
        "days_since_meeting":int(row["days_since_meeting"]),
        "topics":            row["topics"].split(", ") if row["topics"] != "General" else [],
        "word_count":        int(row["note_word_count"]),
    }
    return json.dumps(meta)

df["metadata"] = df.apply(build_metadata, axis=1)

# ── 7. Flag thin notes (word count < 15) ──────────────────────────────────────
THIN_THRESHOLD = 15
df["is_thin_note"] = df["note_word_count"] < THIN_THRESHOLD
thin_count = df["is_thin_note"].sum()
if thin_count:
    print(f"⚠️  {thin_count} notes are below {THIN_THRESHOLD} words — consider enriching or filtering these.")
else:
    print(f"✓ No thin notes found (all notes ≥ {THIN_THRESHOLD} words).")

# ── 8. Drop the temporary parsing column & reorder ────────────────────────────
df.drop(columns=["meeting_date_parsed", "meeting_date_str"], inplace=True)

# Put the two RAG-critical columns first for clarity
priority_cols = ["rag_chunk", "metadata"]
other_cols = [c for c in df.columns if c not in priority_cols]
df = df[priority_cols + other_cols]

# ── 9. Save ────────────────────────────────────────────────────────────────────
df.to_csv(OUTPUT_PATH, index=False)
print(f"\n✓ Saved improved dataset → {OUTPUT_PATH}")
print(f"  Rows: {len(df)}")
print(f"  Columns: {list(df.columns)}")
print(f"\n── Sample rag_chunk ──────────────────────────────────────")
print(df["rag_chunk"].iloc[0])
print(f"\n── Sample metadata ───────────────────────────────────────")
print(df["metadata"].iloc[0])