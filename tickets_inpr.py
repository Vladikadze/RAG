"""
RAG Dataset Improvement Script for support_tickets_cleaned.csv
--------------------------------------------------------------
Issues fixed:
  1. Drops 5 zero-variance bool columns (all True or all False — no signal)
  2. Drops is_low_information_ticket (all False — zero variance)
  3. Drops triple company name columns (keeps company_name only)
  4. Drops ticket_summary_short (100% identical to ticket_text_combined)
  5. Drops ticket_text_combined, support_ticket_text, support_ticket_chunk
     (all near-duplicates — rebuilt cleanly as rag_chunk)
  6. Drops created_month (derivable from created_at)
  7. Drops four redundant word-count columns (keep combined_word_count only)
  8. Flags recycled subjects (15 templates across 80 tickets)
  9. Flags recycled descriptions (16 templates across 80 tickets)
 10. Flags missing resolution (resolution_note == '-', 38 tickets = 48%)
 11. Adds priority_rank (ordinal int) for numeric filtering
 12. Splits into two rag_chunk variants:
     - rag_chunk_full  : includes resolution (for resolved/closed tickets)
     - rag_chunk_issue : subject + description only (for similarity search on open issues)
 13. Adds metadata JSON column for vector DB filtered retrieval
"""

import pandas as pd
import json

INPUT_PATH  = "support_tickets_cleaned.csv"
OUTPUT_PATH = "support_tickets_rag_ready.csv"

# ── 1. Load ────────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_PATH)
print(f"Loaded {len(df)} rows, {df.shape[1]} columns")

# ── 2. Drop redundant / zero-variance columns ──────────────────────────────────
cols_to_drop = [
    # Zero-variance bools (all True)
    "has_customer_id", "has_subject", "has_description",
    "has_resolution_note", "has_created_at",
    # Zero-variance bool (all False)
    "is_low_information_ticket",
    # Redundant company name variants
    "company_name_raw", "company_name_normalized",
    # 100% identical to ticket_text_combined
    "ticket_summary_short",
    # All three are near-duplicates — rebuilt below as rag_chunk
    "ticket_text_combined", "support_ticket_text", "support_ticket_chunk",
    # Derivable from created_at
    "created_month",
    # Redundant word counts (keep combined_word_count)
    "subject_word_count", "description_word_count", "resolution_word_count",
]
df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
print(f"After dropping redundant columns: {df.shape[1]} columns remain")

# ── 3. Parse created_at ────────────────────────────────────────────────────────
df["created_at_str"] = pd.to_datetime(df["created_at"], utc=True).dt.strftime("%B %d, %Y")

# ── 4. Flag recycled subjects ──────────────────────────────────────────────────
subject_freq = df["subject"].value_counts()
recycled_subjects = set(subject_freq[subject_freq >= 3].index)
df["is_recycled_subject"] = df["subject"].isin(recycled_subjects)
print(f"⚠️  {df['is_recycled_subject'].sum()} tickets have recycled subjects "
      f"({len(recycled_subjects)} templates) — retrieval on subject alone will be unreliable.")

# ── 5. Flag recycled descriptions ─────────────────────────────────────────────
desc_freq = df["description"].value_counts()
recycled_descs = set(desc_freq[desc_freq >= 3].index)
df["is_recycled_description"] = df["description"].isin(recycled_descs)
print(f"⚠️  {df['is_recycled_description'].sum()} tickets have recycled descriptions "
      f"({len(recycled_descs)} templates).")

# ── 6. Flag missing resolution notes ──────────────────────────────────────────
df["has_real_resolution"] = df["resolution_note"].str.strip() != "-"
missing_res = (~df["has_real_resolution"]).sum()
print(f"⚠️  {missing_res} tickets ({missing_res/len(df)*100:.0f}%) have no real resolution note ('-'). "
      f"These should not be used as resolved-issue reference chunks.")

# ── 7. Add priority_rank (ordinal) ────────────────────────────────────────────
PRIORITY_ORDER = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
df["priority_rank"] = df["priority"].map(PRIORITY_ORDER)

# ── 8. Build rag_chunk_full (all fields — best for resolved tickets) ───────────
def build_rag_chunk_full(row: pd.Series) -> str:
    res_line = (
        f"Resolution: {row['resolution_note']}"
        if row["has_real_resolution"]
        else "Resolution: [Not yet resolved]"
    )
    subject_flag = " ⚠️ [generic subject]" if row["is_recycled_subject"] else ""
    desc_flag    = " ⚠️ [generic description]" if row["is_recycled_description"] else ""
    lines = [
        f"Support Ticket | {row['ticket_id']} | {row['company_name']} ({row['customer_id']})",
        f"Issue Type: {row['issue_type']} | Priority: {row['priority']} | Status: {row['status']}",
        f"Assigned To: {row['assigned_to']} | Created: {row['created_at_str']}",
        f"Subject: {row['subject']}{subject_flag}",
        f"Description: {row['description']}{desc_flag}",
        res_line,
    ]
    return "\n".join(lines)

df["rag_chunk_full"] = df.apply(build_rag_chunk_full, axis=1)

# ── 9. Build rag_chunk_issue (subject + description only) ─────────────────────
# Use this variant when doing similarity search to find related open issues,
# so resolved tickets don't pollute results with their resolution text.
def build_rag_chunk_issue(row: pd.Series) -> str:
    lines = [
        f"Support Issue | {row['ticket_id']} | {row['company_name']}",
        f"Issue Type: {row['issue_type']} | Priority: {row['priority']}",
        f"Subject: {row['subject']}",
        f"Description: {row['description']}",
    ]
    return "\n".join(lines)

df["rag_chunk_issue"] = df.apply(build_rag_chunk_issue, axis=1)

# ── 10. Build metadata JSON ────────────────────────────────────────────────────
def build_metadata(row: pd.Series) -> str:
    meta = {
        "ticket_id":              row["ticket_id"],
        "customer_id":            row["customer_id"],
        "company_name":           row["company_name"],
        "issue_type":             row["issue_type"],
        "priority":               row["priority"],
        "priority_rank":          int(row["priority_rank"]),
        "status":                 row["status"],
        "assigned_to":            row["assigned_to"],
        "is_resolved":            bool(row["is_resolved"]),
        "has_real_resolution":    bool(row["has_real_resolution"]),
        "risk_score":             float(row["ticket_risk_score"]),
        "is_recycled_subject":    bool(row["is_recycled_subject"]),
        "is_recycled_description":bool(row["is_recycled_description"]),
        "created_at":             row["created_at_str"],
    }
    return json.dumps(meta)

df["metadata"] = df.apply(build_metadata, axis=1)

# ── 11. Drop temp columns & reorder ───────────────────────────────────────────
df.drop(columns=["created_at_str"], inplace=True)

priority_cols = ["rag_chunk_full", "rag_chunk_issue", "metadata"]
other_cols = [c for c in df.columns if c not in priority_cols]
df = df[priority_cols + other_cols]

# ── 12. Save ───────────────────────────────────────────────────────────────────
df.to_csv(OUTPUT_PATH, index=False)
print(f"\n✓ Saved improved dataset → {OUTPUT_PATH}")
print(f"  Rows: {len(df)}")
print(f"  Columns: {list(df.columns)}")
print(f"\n── Sample rag_chunk_full (resolved) ──────────────────────")
resolved_sample = df[df["is_resolved"]].iloc[0]
print(resolved_sample["rag_chunk_full"])
print(f"\n── Sample rag_chunk_full (unresolved) ────────────────────")
open_sample = df[~df["is_resolved"]].iloc[0]
print(open_sample["rag_chunk_full"])
print(f"\n── Sample rag_chunk_issue ────────────────────────────────")
print(open_sample["rag_chunk_issue"])
print(f"\n── Sample metadata ───────────────────────────────────────")
print(df["metadata"].iloc[0])