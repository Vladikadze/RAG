#%%
import pandas as pd
import numpy as np
import re

# -----------------------------
# 1) Load data
# -----------------------------
df = pd.read_csv("/Users/vladislavbogomazov/Desktop/Rag system mvp/data/crm_records/customers.csv")

# -----------------------------
# 2) Standardize column names
# -----------------------------
df.columns = (
    df.columns.str.strip()
              .str.lower()
              .str.replace(r"[^\w]+", "_", regex=True)
              .str.strip("_")
)

# Expected columns
expected_cols = [
    "customer_id",
    "company_name",
    "industry",
    "country",
    "contact_name",
    "contact_email",
    "phone",
    "company_size",
    "current_status",
    "plan_interest",
    "lead_source",
    "created_at",
    "assigned_sales_rep",
]

missing = [c for c in expected_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing expected columns: {missing}")

# -----------------------------
# 3) Basic whitespace cleanup
# -----------------------------
text_cols = [
    "customer_id",
    "company_name",
    "industry",
    "country",
    "contact_name",
    "contact_email",
    "phone",
    "current_status",
    "plan_interest",
    "lead_source",
    "assigned_sales_rep",
]

for col in text_cols:
    df[col] = df[col].astype("string").str.strip()

# Replace empty strings with NA
df = df.replace(r"^\s*$", pd.NA, regex=True)

# -----------------------------
# 4) Helper functions
# -----------------------------
def normalize_email(x):
    if pd.isna(x):
        return pd.NA
    x = str(x).strip().lower()
    return x if "@" in x else pd.NA

def clean_phone(x):
    if pd.isna(x):
        return pd.NA
    x = str(x).strip()
    # keep digits and leading +
    x = re.sub(r"(?!^\+)[^\d]", "", x)
    if x == "":
        return pd.NA
    return x

def normalize_company_name(x):
    if pd.isna(x):
        return pd.NA
    x = str(x).strip().lower()
    x = re.sub(r"[^\w\s]", " ", x)
    x = re.sub(r"\b(inc|llc|ltd|limited|corp|corporation|co|company|gmbh|sa|bv)\b", "", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def clean_title_case(x):
    if pd.isna(x):
        return pd.NA
    return str(x).strip().title()

def normalize_category(x, mapping=None, title_case=False):
    if pd.isna(x):
        return pd.NA
    x = str(x).strip().lower()
    if mapping:
        x = mapping.get(x, x)
    return x.title() if title_case else x

# -----------------------------
# 5) Normalize specific fields
# -----------------------------
df["contact_email"] = df["contact_email"].apply(normalize_email)
df["phone"] = df["phone"].apply(clean_phone)

df["company_name_raw"] = df["company_name"]
df["company_name_normalized"] = df["company_name"].apply(normalize_company_name)

df["company_name"] = df["company_name"].apply(clean_title_case)
df["industry"] = df["industry"].apply(clean_title_case)
df["country"] = df["country"].apply(clean_title_case)
df["contact_name"] = df["contact_name"].apply(clean_title_case)
df["assigned_sales_rep"] = df["assigned_sales_rep"].apply(clean_title_case)

# -----------------------------
# 6) Standardize category values
# -----------------------------
status_map = {
    "active": "Active",
    "inactive": "Inactive",
    "churned": "Churned",
    "pending": "Pending",
    "trial": "Trial",
    "customer": "Customer",
}

lead_source_map = {
    "referral": "Referral",
    "ads": "Ads",
    "ad": "Ads",
    "website": "Website",
    "web": "Website",
    "linkedin": "Linkedin",
    "email campaign": "Email Campaign",
    "campaign": "Email Campaign",
    "event": "Event",
    "outbound": "Outbound",
}

plan_map = {
    "basic": "Basic",
    "pro": "Pro",
    "enterprise": "Enterprise",
    "custom": "Custom",
}

df["current_status"] = df["current_status"].apply(lambda x: normalize_category(x, status_map, title_case=False))
df["lead_source"] = df["lead_source"].apply(lambda x: normalize_category(x, lead_source_map, title_case=False))
df["plan_interest"] = df["plan_interest"].apply(lambda x: normalize_category(x, plan_map, title_case=False))

# -----------------------------
# 7) Fix company_size
# -----------------------------
df["company_size"] = pd.to_numeric(df["company_size"], errors="coerce")

# Optional: flag unrealistic sizes
df.loc[df["company_size"] <= 0, "company_size"] = np.nan

# -----------------------------
# 8) Parse created_at
# -----------------------------
df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)

# -----------------------------
# 9) Remove duplicates
# -----------------------------
# First remove exact duplicate rows
df = df.drop_duplicates()

# Then remove duplicate customer_id, keeping the newest created_at
df = df.sort_values(by="created_at", ascending=False)
df = df.drop_duplicates(subset=["customer_id"], keep="first")

# -----------------------------
# 10) Data quality flags
# -----------------------------
df["has_valid_email"] = df["contact_email"].notna()
df["has_valid_phone"] = df["phone"].notna()
df["is_active_customer"] = df["current_status"].eq("Active")

# Days since created_at
now_utc = pd.Timestamp.utcnow().tz_localize("UTC") if pd.Timestamp.utcnow().tzinfo is None else pd.Timestamp.utcnow()
df["customer_age_days"] = (now_utc - df["created_at"]).dt.days

# -----------------------------
# 11) Create a search text field for RAG / retrieval
# -----------------------------
df["customer_search_text"] = (
    "Customer ID: " + df["customer_id"].fillna("").astype(str) + " | " +
    "Company: " + df["company_name"].fillna("").astype(str) + " | " +
    "Industry: " + df["industry"].fillna("").astype(str) + " | " +
    "Country: " + df["country"].fillna("").astype(str) + " | " +
    "Contact: " + df["contact_name"].fillna("").astype(str) + " | " +
    "Email: " + df["contact_email"].fillna("").astype(str) + " | " +
    "Company Size: " + df["company_size"].fillna(-1).astype(int).astype(str) + " | " +
    "Status: " + df["current_status"].fillna("").astype(str) + " | " +
    "Plan Interest: " + df["plan_interest"].fillna("").astype(str) + " | " +
    "Lead Source: " + df["lead_source"].fillna("").astype(str) + " | " +
    "Assigned Sales Rep: " + df["assigned_sales_rep"].fillna("").astype(str)
)

# -----------------------------
# 12) Optional: reorder columns
# -----------------------------
final_cols = [
    "customer_id",
    "company_name",
    "company_name_raw",
    "company_name_normalized",
    "industry",
    "country",
    "contact_name",
    "contact_email",
    "phone",
    "company_size",
    "current_status",
    "plan_interest",
    "lead_source",
    "assigned_sales_rep",
    "created_at",
    "has_valid_email",
    "has_valid_phone",
    "is_active_customer",
    "customer_age_days",
    "customer_search_text",
]

df = df[[c for c in final_cols if c in df.columns]]

# -----------------------------
# 13) Save cleaned data
# -----------------------------
df.to_csv("customers_cleaned.csv", index=False)

print("Cleaning complete.")
print(f"Rows after cleaning: {len(df)}")
print("\nMissing values:")
print(df.isna().sum())
print("\nSample:")
print(df.head())
#%%
import pandas as pd
import numpy as np
import re

# -----------------------------
# 1) Load data
# -----------------------------
df = pd.read_csv("/Users/vladislavbogomazov/Desktop/Rag system mvp/data/crm_records/leads.csv")

# -----------------------------
# 2) Standardize column names
# -----------------------------
df.columns = (
    df.columns.str.strip()
              .str.lower()
              .str.replace(r"[^\w]+", "_", regex=True)
              .str.strip("_")
)

# Expected columns
expected_cols = [
    "lead_id",
    "company_name",
    "contact_name",
    "email",
    "interest_area",
    "budget_range",
    "urgency",
    "lead_source",
    "status",
    "notes",
]

missing = [c for c in expected_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing expected columns: {missing}")

# -----------------------------
# 3) Basic whitespace cleanup
# -----------------------------
text_cols = [
    "lead_id",
    "company_name",
    "contact_name",
    "email",
    "interest_area",
    "budget_range",
    "urgency",
    "lead_source",
    "status",
    "notes",
]

for col in text_cols:
    df[col] = df[col].astype("string").str.strip()

# Replace empty strings with NA
df = df.replace(r"^\s*$", pd.NA, regex=True)

# -----------------------------
# 4) Helper functions
# -----------------------------
def normalize_email(x):
    if pd.isna(x):
        return pd.NA
    x = str(x).strip().lower()
    return x if "@" in x else pd.NA

def normalize_company_name(x):
    if pd.isna(x):
        return pd.NA
    x = str(x).strip().lower()
    x = re.sub(r"[^\w\s]", " ", x)
    x = re.sub(r"\b(inc|llc|ltd|limited|corp|corporation|co|company|gmbh|sa|bv)\b", "", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def clean_title_case(x):
    if pd.isna(x):
        return pd.NA
    return str(x).strip().title()

def clean_notes(x):
    if pd.isna(x):
        return pd.NA
    x = str(x)
    x = re.sub(r"\s+", " ", x).strip()
    return x if x else pd.NA

def normalize_category(x, mapping=None):
    if pd.isna(x):
        return pd.NA
    x = str(x).strip().lower()
    if mapping:
        x = mapping.get(x, x)
    return x

# -----------------------------
# 5) Normalize specific fields
# -----------------------------
df["email"] = df["email"].apply(normalize_email)

df["company_name_raw"] = df["company_name"]
df["company_name_normalized"] = df["company_name"].apply(normalize_company_name)

df["company_name"] = df["company_name"].apply(clean_title_case)
df["contact_name"] = df["contact_name"].apply(clean_title_case)
df["interest_area"] = df["interest_area"].apply(clean_title_case)
df["notes"] = df["notes"].apply(clean_notes)

# -----------------------------
# 6) Standardize category values
# -----------------------------
urgency_map = {
    "low": "Low",
    "medium": "Medium",
    "med": "Medium",
    "high": "High",
    "urgent": "High",
}

status_map = {
    "new": "New",
    "open": "Open",
    "contacted": "Contacted",
    "qualified": "Qualified",
    "proposal": "Proposal",
    "negotiation": "Negotiation",
    "won": "Won",
    "lost": "Lost",
    "closed": "Closed",
    "disqualified": "Disqualified",
}

lead_source_map = {
    "referral": "Referral",
    "ads": "Ads",
    "ad": "Ads",
    "website": "Website",
    "web": "Website",
    "linkedin": "Linkedin",
    "email campaign": "Email Campaign",
    "campaign": "Email Campaign",
    "event": "Event",
    "outbound": "Outbound",
    "partner": "Partner",
}

budget_map = {
    "<10k": "<10K",
    "under 10k": "<10K",
    "10k-25k": "10K-25K",
    "25k-50k": "25K-50K",
    "50k-100k": "50K-100K",
    "100k+": "100K+",
    ">100k": "100K+",
}

df["urgency"] = df["urgency"].apply(lambda x: normalize_category(x, urgency_map))
df["status"] = df["status"].apply(lambda x: normalize_category(x, status_map))
df["lead_source"] = df["lead_source"].apply(lambda x: normalize_category(x, lead_source_map))
df["budget_range"] = df["budget_range"].apply(lambda x: normalize_category(x, budget_map))

# Convert normalized categories to title case where useful
for col in ["urgency", "status", "lead_source", "budget_range"]:
    df[col] = df[col].apply(lambda x: x.title() if pd.notna(x) else pd.NA)

# -----------------------------
# 7) Remove duplicates
# -----------------------------
# Remove exact duplicate rows
df = df.drop_duplicates()

# Keep best row per lead_id
# Preference: rows with more non-null fields
df["_non_null_count"] = df.notna().sum(axis=1)
df = df.sort_values(by="_non_null_count", ascending=False)
df = df.drop_duplicates(subset=["lead_id"], keep="first")

# -----------------------------
# 8) Optional lead matching helpers
# -----------------------------
df["email_domain"] = df["email"].str.extract(r"@(.+)$", expand=False)

# -----------------------------
# 9) Lead quality scoring
# -----------------------------
urgency_score = {
    "Low": 1,
    "Medium": 2,
    "High": 3,
}

budget_score = {
    "<10K": 1,
    "10K-25K": 2,
    "25K-50K": 3,
    "50K-100K": 4,
    "100K+": 5,
}

status_score = {
    "New": 1,
    "Open": 1,
    "Contacted": 2,
    "Qualified": 3,
    "Proposal": 4,
    "Negotiation": 5,
    "Won": 5,
    "Closed": 2,
    "Lost": 0,
    "Disqualified": 0,
}

df["lead_quality_score"] = (
    df["urgency"].map(urgency_score).fillna(0)
    + df["budget_range"].map(budget_score).fillna(0)
    + df["status"].map(status_score).fillna(0)
    + df["notes"].notna().astype(int)
    + df["email"].notna().astype(int)
)

# Optional label
df["lead_quality_band"] = pd.cut(
    df["lead_quality_score"],
    bins=[-1, 3, 6, 10, 20],
    labels=["Low", "Medium", "High", "Very High"]
)

# -----------------------------
# 10) Data quality flags
# -----------------------------
df["has_valid_email"] = df["email"].notna()
df["has_notes"] = df["notes"].notna()

# -----------------------------
# 11) Create retrieval/search text for RAG
# -----------------------------
df["lead_search_text"] = (
    "Lead ID: " + df["lead_id"].fillna("").astype(str) + " | " +
    "Company: " + df["company_name"].fillna("").astype(str) + " | " +
    "Contact: " + df["contact_name"].fillna("").astype(str) + " | " +
    "Email: " + df["email"].fillna("").astype(str) + " | " +
    "Interest Area: " + df["interest_area"].fillna("").astype(str) + " | " +
    "Budget Range: " + df["budget_range"].fillna("").astype(str) + " | " +
    "Urgency: " + df["urgency"].fillna("").astype(str) + " | " +
    "Lead Source: " + df["lead_source"].fillna("").astype(str) + " | " +
    "Status: " + df["status"].fillna("").astype(str) + " | " +
    "Notes: " + df["notes"].fillna("").astype(str)
)

# -----------------------------
# 12) Create a profile text for semantic retrieval
# -----------------------------
df["lead_profile_text"] = (
    "Lead Profile\n"
    + "Lead ID: " + df["lead_id"].fillna("").astype(str) + "\n"
    + "Company: " + df["company_name"].fillna("").astype(str) + "\n"
    + "Contact Name: " + df["contact_name"].fillna("").astype(str) + "\n"
    + "Email: " + df["email"].fillna("").astype(str) + "\n"
    + "Interest Area: " + df["interest_area"].fillna("").astype(str) + "\n"
    + "Budget Range: " + df["budget_range"].fillna("").astype(str) + "\n"
    + "Urgency: " + df["urgency"].fillna("").astype(str) + "\n"
    + "Lead Source: " + df["lead_source"].fillna("").astype(str) + "\n"
    + "Status: " + df["status"].fillna("").astype(str) + "\n"
    + "Notes: " + df["notes"].fillna("").astype(str) + "\n"
    + "Lead Quality Score: " + df["lead_quality_score"].fillna(0).astype(int).astype(str)
)

# -----------------------------
# 13) Reorder columns
# -----------------------------
final_cols = [
    "lead_id",
    "company_name",
    "company_name_raw",
    "company_name_normalized",
    "contact_name",
    "email",
    "email_domain",
    "interest_area",
    "budget_range",
    "urgency",
    "lead_source",
    "status",
    "notes",
    "has_valid_email",
    "has_notes",
    "lead_quality_score",
    "lead_quality_band",
    "lead_search_text",
    "lead_profile_text",
]

df = df[[c for c in final_cols if c in df.columns]]

# Drop helper col if still present
if "_non_null_count" in df.columns:
    df = df.drop(columns="_non_null_count")

# -----------------------------
# 14) Save cleaned data
# -----------------------------
df.to_csv("leads_cleaned.csv", index=False)

print("Cleaning complete.")
print(f"Rows after cleaning: {len(df)}")
print("\nMissing values:")
print(df.isna().sum())
print("\nSample:")
print(df.head())
# %%
import pandas as pd
import numpy as np
import re

# -----------------------------
# 1) Load data
# -----------------------------
df = pd.read_csv("/Users/vladislavbogomazov/Desktop/Rag system mvp/data/sales/sales_notes.csv")

# -----------------------------
# 2) Standardize column names
# -----------------------------
df.columns = (
    df.columns.str.strip()
              .str.lower()
              .str.replace(r"[^\w]+", "_", regex=True)
              .str.strip("_")
)

# Expected columns
expected_cols = [
    "note_id",
    "customer_id",
    "company_name",
    "sales_rep",
    "meeting_date",
    "note_text",
]

missing = [c for c in expected_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing expected columns: {missing}")

# -----------------------------
# 3) Basic whitespace cleanup
# -----------------------------
text_cols = [
    "note_id",
    "customer_id",
    "company_name",
    "sales_rep",
    "note_text",
]

for col in text_cols:
    df[col] = df[col].astype("string").str.strip()

# Replace empty strings with NA
df = df.replace(r"^\s*$", pd.NA, regex=True)

# -----------------------------
# 4) Helper functions
# -----------------------------
def normalize_company_name(x):
    if pd.isna(x):
        return pd.NA
    x = str(x).strip().lower()
    x = re.sub(r"[^\w\s]", " ", x)
    x = re.sub(r"\b(inc|llc|ltd|limited|corp|corporation|co|company|gmbh|sa|bv)\b", "", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x if x else pd.NA

def clean_title_case(x):
    if pd.isna(x):
        return pd.NA
    x = str(x).strip()
    return x.title() if x else pd.NA

def clean_note_text(x):
    if pd.isna(x):
        return pd.NA
    x = str(x)

    # Normalize line breaks and spaces
    x = x.replace("\r", "\n")
    x = re.sub(r"\n+", "\n", x)
    x = re.sub(r"[ \t]+", " ", x)

    # Remove very common boilerplate-ish prefixes if present
    x = re.sub(r"^(notes?|meeting notes?)\s*[:\-]\s*", "", x, flags=re.IGNORECASE)

    # Strip repeated separators
    x = re.sub(r"[-_=]{3,}", " ", x)

    # Final cleanup
    x = re.sub(r"\s+", " ", x).strip()

    return x if x else pd.NA

def short_summary(text, max_words=40):
    if pd.isna(text):
        return pd.NA
    words = str(text).split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + "..."

# -----------------------------
# 5) Normalize fields
# -----------------------------
df["company_name_raw"] = df["company_name"]
df["company_name_normalized"] = df["company_name"].apply(normalize_company_name)

df["company_name"] = df["company_name"].apply(clean_title_case)
df["sales_rep"] = df["sales_rep"].apply(clean_title_case)
df["note_text"] = df["note_text"].apply(clean_note_text)

# -----------------------------
# 6) Parse dates
# -----------------------------
df["meeting_date"] = pd.to_datetime(df["meeting_date"], errors="coerce", utc=True)

# -----------------------------
# 7) Remove duplicates
# -----------------------------
# Exact duplicate rows
df = df.drop_duplicates()

# Prefer the row with more filled values, then latest meeting_date
df["_non_null_count"] = df.notna().sum(axis=1)
df = df.sort_values(by=["_non_null_count", "meeting_date"], ascending=[False, False])

# One row per note_id
df = df.drop_duplicates(subset=["note_id"], keep="first")

# -----------------------------
# 8) Quality flags
# -----------------------------
df["has_customer_id"] = df["customer_id"].notna()
df["has_note_text"] = df["note_text"].notna()
df["has_meeting_date"] = df["meeting_date"].notna()

# -----------------------------
# 9) Useful note metrics
# -----------------------------
df["note_word_count"] = df["note_text"].fillna("").astype(str).str.split().str.len()

now_utc = pd.Timestamp.utcnow().tz_localize("UTC") if pd.Timestamp.utcnow().tzinfo is None else pd.Timestamp.utcnow()
df["days_since_meeting"] = (now_utc - df["meeting_date"]).dt.days

# Optional info-density flag
df["is_low_information_note"] = df["note_word_count"].fillna(0) < 8

# -----------------------------
# 10) Short retrieval-friendly summary
# -----------------------------
df["note_summary_short"] = df["note_text"].apply(lambda x: short_summary(x, max_words=40))

# -----------------------------
# 11) Build lexical search text
# -----------------------------
df["sales_note_text"] = (
    "Note ID: " + df["note_id"].fillna("").astype(str) + " | " +
    "Customer ID: " + df["customer_id"].fillna("").astype(str) + " | " +
    "Company: " + df["company_name"].fillna("").astype(str) + " | " +
    "Sales Rep: " + df["sales_rep"].fillna("").astype(str) + " | " +
    "Meeting Date: " + df["meeting_date"].astype("string").fillna("") + " | " +
    "Summary: " + df["note_summary_short"].fillna("").astype(str) + " | " +
    "Note: " + df["note_text"].fillna("").astype(str)
)

# -----------------------------
# 12) Build semantic retrieval chunk
# -----------------------------
df["sales_note_chunk"] = (
    "Sales Note\n"
    + "Note ID: " + df["note_id"].fillna("").astype(str) + "\n"
    + "Customer ID: " + df["customer_id"].fillna("").astype(str) + "\n"
    + "Company: " + df["company_name"].fillna("").astype(str) + "\n"
    + "Sales Rep: " + df["sales_rep"].fillna("").astype(str) + "\n"
    + "Meeting Date: " + df["meeting_date"].astype("string").fillna("") + "\n"
    + "Short Summary: " + df["note_summary_short"].fillna("").astype(str) + "\n"
    + "Full Note: " + df["note_text"].fillna("").astype(str)
)

# -----------------------------
# 13) Optional grouped key for later rollups
# -----------------------------
df["meeting_month"] = df["meeting_date"].dt.to_period("M").astype("string")

# -----------------------------
# 14) Reorder columns
# -----------------------------
final_cols = [
    "note_id",
    "customer_id",
    "company_name",
    "company_name_raw",
    "company_name_normalized",
    "sales_rep",
    "meeting_date",
    "meeting_month",
    "note_text",
    "note_summary_short",
    "note_word_count",
    "days_since_meeting",
    "has_customer_id",
    "has_note_text",
    "has_meeting_date",
    "is_low_information_note",
    "sales_note_text",
    "sales_note_chunk",
]

df = df[[c for c in final_cols if c in df.columns]]

# Drop helper if still present
if "_non_null_count" in df.columns:
    df = df.drop(columns="_non_null_count")

# -----------------------------
# 15) Save cleaned file
# -----------------------------
df.to_csv("sales_notes_cleaned.csv", index=False)

print("Cleaning complete.")
print(f"Rows after cleaning: {len(df)}")
print("\nMissing values:")
print(df.isna().sum())
print("\nSample:")
print(df.head())
# %%

import pandas as pd
import numpy as np
import re

# -----------------------------
# 1) Load data
# -----------------------------
df = pd.read_csv("/Users/vladislavbogomazov/Desktop/Rag system mvp/data/tickets/support_tickets_80.csv")

# -----------------------------
# 2) Standardize column names
# -----------------------------
df.columns = (
    df.columns.str.strip()
              .str.lower()
              .str.replace(r"[^\w]+", "_", regex=True)
              .str.strip("_")
)

# Expected columns
expected_cols = [
    "ticket_id",
    "customer_id",
    "company_name",
    "issue_type",
    "priority",
    "subject",
    "description",
    "status",
    "created_at",
    "assigned_to",
    "resolution_note",
]

missing = [c for c in expected_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing expected columns: {missing}")

# -----------------------------
# 3) Basic whitespace cleanup
# -----------------------------
text_cols = [
    "ticket_id",
    "customer_id",
    "company_name",
    "issue_type",
    "priority",
    "subject",
    "description",
    "status",
    "assigned_to",
    "resolution_note",
]

for col in text_cols:
    df[col] = df[col].astype("string").str.strip()

# Replace empty strings with NA
df = df.replace(r"^\s*$", pd.NA, regex=True)

# -----------------------------
# 4) Helper functions
# -----------------------------
def normalize_company_name(x):
    if pd.isna(x):
        return pd.NA
    x = str(x).strip().lower()
    x = re.sub(r"[^\w\s]", " ", x)
    x = re.sub(r"\b(inc|llc|ltd|limited|corp|corporation|co|company|gmbh|sa|bv)\b", "", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x if x else pd.NA

def clean_title_case(x):
    if pd.isna(x):
        return pd.NA
    x = str(x).strip()
    return x.title() if x else pd.NA

def clean_free_text(x):
    if pd.isna(x):
        return pd.NA
    x = str(x)
    x = x.replace("\r", "\n")
    x = re.sub(r"\n+", "\n", x)
    x = re.sub(r"[ \t]+", " ", x)
    x = re.sub(r"[-_=]{3,}", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x if x else pd.NA

def short_summary(text, max_words=40):
    if pd.isna(text):
        return pd.NA
    words = str(text).split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + "..."

def normalize_category(x, mapping=None):
    if pd.isna(x):
        return pd.NA
    x = str(x).strip().lower()
    if mapping:
        x = mapping.get(x, x)
    return x

# -----------------------------
# 5) Normalize core fields
# -----------------------------
df["company_name_raw"] = df["company_name"]
df["company_name_normalized"] = df["company_name"].apply(normalize_company_name)

df["company_name"] = df["company_name"].apply(clean_title_case)
df["assigned_to"] = df["assigned_to"].apply(clean_title_case)

df["subject"] = df["subject"].apply(clean_free_text)
df["description"] = df["description"].apply(clean_free_text)
df["resolution_note"] = df["resolution_note"].apply(clean_free_text)
df["issue_type"] = df["issue_type"].apply(clean_title_case)

# -----------------------------
# 6) Standardize categories
# -----------------------------
priority_map = {
    "low": "Low",
    "medium": "Medium",
    "med": "Medium",
    "high": "High",
    "urgent": "Urgent",
    "critical": "Critical",
}

status_map = {
    "open": "Open",
    "new": "New",
    "in progress": "In Progress",
    "in_progress": "In Progress",
    "pending": "Pending",
    "resolved": "Resolved",
    "closed": "Closed",
    "reopened": "Reopened",
    "on hold": "On Hold",
    "on_hold": "On Hold",
}

issue_type_map = {
    "billing": "Billing",
    "api": "Api",
    "integration": "Integration",
    "login": "Login",
    "authentication": "Authentication",
    "onboarding": "Onboarding",
    "bug": "Bug",
    "performance": "Performance",
    "security": "Security",
    "account": "Account",
    "ui": "Ui",
    "feature request": "Feature Request",
    "feature_request": "Feature Request",
}

df["priority"] = df["priority"].apply(lambda x: normalize_category(x, priority_map))
df["status"] = df["status"].apply(lambda x: normalize_category(x, status_map))
df["issue_type"] = df["issue_type"].apply(lambda x: normalize_category(x, issue_type_map))

for col in ["priority", "status", "issue_type"]:
    df[col] = df[col].apply(lambda x: x.title() if pd.notna(x) else pd.NA)

# -----------------------------
# 7) Parse created_at
# -----------------------------
df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)

# -----------------------------
# 8) Remove duplicates
# -----------------------------
# Remove exact duplicates
df = df.drop_duplicates()

# Prefer rows with more filled values, then latest created_at
df["_non_null_count"] = df.notna().sum(axis=1)
df = df.sort_values(by=["_non_null_count", "created_at"], ascending=[False, False])

# Keep one row per ticket_id
df = df.drop_duplicates(subset=["ticket_id"], keep="first")

# -----------------------------
# 9) Build combined ticket text
# -----------------------------
df["ticket_text_combined"] = (
    "Subject: " + df["subject"].fillna("").astype(str) + " | " +
    "Description: " + df["description"].fillna("").astype(str) + " | " +
    "Resolution: " + df["resolution_note"].fillna("").astype(str)
)

# -----------------------------
# 10) Quality flags
# -----------------------------
df["has_customer_id"] = df["customer_id"].notna()
df["has_subject"] = df["subject"].notna()
df["has_description"] = df["description"].notna()
df["has_resolution_note"] = df["resolution_note"].notna()
df["has_created_at"] = df["created_at"].notna()
df["is_resolved"] = df["status"].isin(["Resolved", "Closed"])

# -----------------------------
# 11) Useful ticket metrics
# -----------------------------
df["subject_word_count"] = df["subject"].fillna("").astype(str).str.split().str.len()
df["description_word_count"] = df["description"].fillna("").astype(str).str.split().str.len()
df["resolution_word_count"] = df["resolution_note"].fillna("").astype(str).str.split().str.len()
df["combined_word_count"] = df["ticket_text_combined"].fillna("").astype(str).str.split().str.len()

now_utc = pd.Timestamp.utcnow().tz_localize("UTC") if pd.Timestamp.utcnow().tzinfo is None else pd.Timestamp.utcnow()
df["days_since_created"] = (now_utc - df["created_at"]).dt.days

df["is_low_information_ticket"] = (
    df["subject_word_count"].fillna(0)
    + df["description_word_count"].fillna(0)
) < 8

# -----------------------------
# 12) Priority / risk helpers
# -----------------------------
priority_score = {
    "Low": 1,
    "Medium": 2,
    "High": 3,
    "Urgent": 4,
    "Critical": 5,
}

status_score = {
    "New": 2,
    "Open": 2,
    "In Progress": 2,
    "Pending": 1,
    "Resolved": 0,
    "Closed": 0,
    "Reopened": 3,
    "On Hold": 1,
}

df["ticket_risk_score"] = (
    df["priority"].map(priority_score).fillna(0)
    + df["status"].map(status_score).fillna(0)
    + (~df["is_resolved"]).astype(int)
)

# -----------------------------
# 13) Short summary
# -----------------------------
df["ticket_summary_short"] = df["ticket_text_combined"].apply(lambda x: short_summary(x, max_words=45))

# -----------------------------
# 14) Lexical retrieval text
# -----------------------------
df["support_ticket_text"] = (
    "Ticket ID: " + df["ticket_id"].fillna("").astype(str) + " | " +
    "Customer ID: " + df["customer_id"].fillna("").astype(str) + " | " +
    "Company: " + df["company_name"].fillna("").astype(str) + " | " +
    "Issue Type: " + df["issue_type"].fillna("").astype(str) + " | " +
    "Priority: " + df["priority"].fillna("").astype(str) + " | " +
    "Status: " + df["status"].fillna("").astype(str) + " | " +
    "Created At: " + df["created_at"].astype("string").fillna("") + " | " +
    "Assigned To: " + df["assigned_to"].fillna("").astype(str) + " | " +
    "Summary: " + df["ticket_summary_short"].fillna("").astype(str) + " | " +
    "Details: " + df["ticket_text_combined"].fillna("").astype(str)
)

# -----------------------------
# 15) Semantic retrieval chunk
# -----------------------------
df["support_ticket_chunk"] = (
    "Support Ticket\n"
    + "Ticket ID: " + df["ticket_id"].fillna("").astype(str) + "\n"
    + "Customer ID: " + df["customer_id"].fillna("").astype(str) + "\n"
    + "Company: " + df["company_name"].fillna("").astype(str) + "\n"
    + "Issue Type: " + df["issue_type"].fillna("").astype(str) + "\n"
    + "Priority: " + df["priority"].fillna("").astype(str) + "\n"
    + "Status: " + df["status"].fillna("").astype(str) + "\n"
    + "Created At: " + df["created_at"].astype("string").fillna("") + "\n"
    + "Assigned To: " + df["assigned_to"].fillna("").astype(str) + "\n"
    + "Short Summary: " + df["ticket_summary_short"].fillna("").astype(str) + "\n"
    + "Subject: " + df["subject"].fillna("").astype(str) + "\n"
    + "Description: " + df["description"].fillna("").astype(str) + "\n"
    + "Resolution Note: " + df["resolution_note"].fillna("").astype(str)
)

# -----------------------------
# 16) Useful grouping field
# -----------------------------
df["created_month"] = df["created_at"].dt.to_period("M").astype("string")

# -----------------------------
# 17) Reorder columns
# -----------------------------
final_cols = [
    "ticket_id",
    "customer_id",
    "company_name",
    "company_name_raw",
    "company_name_normalized",
    "issue_type",
    "priority",
    "status",
    "assigned_to",
    "created_at",
    "created_month",
    "subject",
    "description",
    "resolution_note",
    "ticket_text_combined",
    "ticket_summary_short",
    "subject_word_count",
    "description_word_count",
    "resolution_word_count",
    "combined_word_count",
    "days_since_created",
    "has_customer_id",
    "has_subject",
    "has_description",
    "has_resolution_note",
    "has_created_at",
    "is_resolved",
    "is_low_information_ticket",
    "ticket_risk_score",
    "support_ticket_text",
    "support_ticket_chunk",
]

df = df[[c for c in final_cols if c in df.columns]]

if "_non_null_count" in df.columns:
    df = df.drop(columns="_non_null_count")

# -----------------------------
# 18) Save cleaned data
# -----------------------------
df.to_csv("support_tickets_cleaned.csv", index=False)

print("Cleaning complete.")
print(f"Rows after cleaning: {len(df)}")
print("\nMissing values:")
print(df.isna().sum())
print("\nSample:")
print(df.head())
# %%
