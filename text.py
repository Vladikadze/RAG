import re
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any


INPUT_DIR = Path("/Users/vladislavbogomazov/Desktop/Rag system mvp/data")
OUTPUT_FILE = Path("rag_chunks_txt.jsonl")


# -----------------------------
# Generic helpers
# -----------------------------
def stable_chunk_id(source_file: str, part_name: str, text: str) -> str:
    raw = f"{source_file}::{part_name}::{text}"
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
    return f"{source_file}::{part_name}::{digest}"


def clean_text(text: str) -> str:
    text = text.replace("\u2011", "-")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def write_jsonl(rows: List[Dict[str, Any]], output_file: Path) -> None:
    with output_file.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# -----------------------------
# File type detection
# -----------------------------
def detect_file_type(file_path: Path, text: str) -> str:
    name = file_path.name.lower()

    if "meeting_note_" in name:
        return "meeting_note"

    if re.search(r"Q:\s*.*?\nA:\s*", text, re.DOTALL | re.IGNORECASE):
        return "faq"

    return "document"


# -----------------------------
# FAQ parsing
# -----------------------------
def infer_faq_type(filename: str) -> str:
    name = filename.lower()

    if "pricing" in name:
        return "pricing"
    if "implementation" in name:
        return "implementation"
    if "integration" in name:
        return "integrations"
    if "general" in name:
        return "general"

    return "unknown"


def parse_qa_pairs(text: str) -> List[Dict[str, str]]:
    pattern = re.compile(
        r"Q:\s*(.*?)\s*A:\s*(.*?)(?=\nQ:\s*|\Z)",
        re.DOTALL | re.IGNORECASE
    )

    pairs = []
    for match in pattern.finditer(text):
        question = match.group(1).strip()
        answer = match.group(2).strip()
        if question and answer:
            pairs.append({"question": question, "answer": answer})

    return pairs


def process_faq_file(file_path: Path, text: str) -> List[Dict[str, Any]]:
    faq_type = infer_faq_type(file_path.name)
    qa_pairs = parse_qa_pairs(text)
    chunks = []

    for pair in qa_pairs:
        chunk_text = f"Question: {pair['question']}\nAnswer: {pair['answer']}"
        metadata = {
            "record_type": "faq",
            "faq_type": faq_type,
            "source_file": file_path.name,
            "source_path": str(file_path),
            "question": pair["question"]
        }

        chunks.append({
            "id": stable_chunk_id(file_path.name, "faq", chunk_text),
            "text": chunk_text,
            "metadata": metadata
        })

    return chunks


# -----------------------------
# Document parsing
# -----------------------------
def infer_doc_type(filename: str) -> str:
    name = filename.lower()

    if "policy" in name:
        return "policy"
    if "proposal" in name:
        return "proposal"
    if "feature_sheet" in name:
        return "feature_sheet"
    if "platform" in name:
        return "product"
    if "integration_guide" in name:
        return "integration_guide"
    if "onboarding_checklist" in name:
        return "checklist"
    if "onboarding_process" in name:
        return "process"

    return "document"


def infer_domain(filename: str) -> str:
    name = filename.lower()

    if "ai_agents" in name:
        return "ai_agents"
    if "allmessage" in name:
        return "allmessage"
    if "argo_engine" in name:
        return "argo_engine"
    if "crm" in name:
        return "crm"
    if "support" in name or "escalation" in name:
        return "support"
    if "refund" in name or "pricing" in name:
        return "commercial"
    if "security" in name:
        return "security"
    if "integration" in name:
        return "integration"
    if "onboarding" in name:
        return "onboarding"

    return "general"


def extract_title(text: str, fallback_filename: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return fallback_filename

    first = lines[0]
    if ":" in first and len(first) < 120:
        return first

    return fallback_filename.replace(".txt", "").replace("_", " ").title()


def normalize_heading(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip().rstrip(":")).strip()


def is_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False

    known_headings = {
        "Overview",
        "Core Capabilities",
        "Key Features",
        "Typical Use Cases",
        "Benefits",
        "Integration Options",
        "Limitations and Considerations",
        "Supported Integration Areas",
        "Integration Approach",
        "Typical Steps",
        "Common Challenges",
        "Client Objective",
        "Proposed Scope",
        "Delivery Phases",
        "Estimated Timeline",
        "Key Benefits",
        "Risks and Considerations",
        "Pre-Go-Live Checklist",
        "Post-Go-Live Checklist",
        "Response Time Targets",
        "Support Availability",
        "Escalation",
        "Data Protection Principles",
        "Data Handling",
        "Security Reviews",
        "Incident Response",
        "Subscription Services",
        "Implementation Projects",
        "Exceptions",
        "Main Features",
        "Example Use Cases",
        "Operational Notes",
        "Onboarding Steps",
        "Escalation Levels",
        "Communication",
    }

    if stripped.rstrip(":") in known_headings:
        return True

    heading_patterns = [
        r"^[A-Z][A-Za-z0-9 /&()\-]+:$",
        r"^\d+\.\s+[A-Z][A-Za-z0-9 /&()\-]+$",
        r"^[A-Z][A-Za-z0-9 /&()\-]+$"
    ]

    return any(re.match(pattern, stripped) for pattern in heading_patterns)


def split_into_sections(text: str) -> List[Dict[str, str]]:
    lines = text.splitlines()
    sections = []

    current_heading = "Document Summary"
    current_content = []

    for line in lines:
        if is_heading(line):
            if current_content:
                content = "\n".join(current_content).strip()
                if content:
                    sections.append({
                        "section_title": current_heading,
                        "section_text": content
                    })
            current_heading = normalize_heading(line)
            current_content = []
        else:
            current_content.append(line)

    if current_content:
        content = "\n".join(current_content).strip()
        if content:
            sections.append({
                "section_title": current_heading,
                "section_text": content
            })

    return sections


def process_document_file(file_path: Path, text: str) -> List[Dict[str, Any]]:
    title = extract_title(text, file_path.name)
    doc_type = infer_doc_type(file_path.name)
    domain = infer_domain(file_path.name)

    sections = split_into_sections(text)
    chunks = []

    for section in sections:
        section_text = section["section_text"].strip()
        if len(section_text) < 20:
            continue

        chunk_text = (
            f"Document: {title}\n"
            f"Section: {section['section_title']}\n\n"
            f"{section_text}"
        )

        metadata = {
            "record_type": "document",
            "doc_type": doc_type,
            "domain": domain,
            "source_file": file_path.name,
            "source_path": str(file_path),
            "document_title": title,
            "section_title": section["section_title"]
        }

        chunks.append({
            "id": stable_chunk_id(file_path.name, section["section_title"], chunk_text),
            "text": chunk_text,
            "metadata": metadata
        })

    if not chunks:
        chunk_text = f"Document: {title}\nSection: Full Document\n\n{text}"
        metadata = {
            "record_type": "document",
            "doc_type": doc_type,
            "domain": domain,
            "source_file": file_path.name,
            "source_path": str(file_path),
            "document_title": title,
            "section_title": "Full Document"
        }

        chunks.append({
            "id": stable_chunk_id(file_path.name, "full_document", chunk_text),
            "text": chunk_text,
            "metadata": metadata
        })

    return chunks


# -----------------------------
# Meeting note parsing
# -----------------------------
def extract_field(text: str, field_name: str) -> str:
    pattern = rf"^{re.escape(field_name)}:\s*(.+)$"
    match = re.search(pattern, text, re.MULTILINE)
    return match.group(1).strip() if match else ""


def extract_bullet_block(text: str, heading: str) -> List[str]:
    pattern = rf"{re.escape(heading)}:\s*(.*?)(?=\n[A-Z][A-Za-z ]+:\s|\Z)"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return []

    block = match.group(1).strip()
    lines = []
    for line in block.splitlines():
        line = line.strip()
        if line.startswith("-"):
            lines.append(line.lstrip("- ").strip())
    return lines


def infer_meeting_topics(summary_items: List[str], next_steps: List[str]) -> List[str]:
    text = " ".join(summary_items + next_steps).lower()
    topics = []

    topic_keywords = {
        "whatsapp_integration": ["whatsapp integration"],
        "api_integration": ["api integration"],
        "dashboard_reporting": ["dashboard reporting"],
        "data_migration": ["data migration"],
        "support_sla": ["support sla"],
        "document_automation": ["document automation"],
        "crm_workflow_automation": ["crm workflow automation"],
        "branch_permissions": ["branch-level permissions"],
        "lead_assignment": ["lead assignment"],
        "ai_chatbot": ["ai chatbot deployment"],
        "security_compliance": ["security and compliance"],
        "pilot_scope": ["pilot scope"],
        "onboarding": ["onboarding"],
        "implementation_timeline": ["implementation timeline", "delivery expectations"],
        "pricing": ["pricing broken down by module", "pricing"]
    }

    for topic, keywords in topic_keywords.items():
        if any(keyword in text for keyword in keywords):
            topics.append(topic)

    return sorted(set(topics))


def process_meeting_note_file(file_path: Path, text: str) -> List[Dict[str, Any]]:
    client = extract_field(text, "Client")
    customer_id = extract_field(text, "Customer ID")
    date_value = extract_field(text, "Date")
    industry = extract_field(text, "Industry")
    country = extract_field(text, "Country")
    attendees_raw = extract_field(text, "Attendees")

    attendees = [a.strip() for a in attendees_raw.split(",") if a.strip()]
    summary_items = extract_bullet_block(text, "Meeting Summary")
    next_steps = extract_bullet_block(text, "Next Steps")

    main_concern = ""
    general_impression = ""
    notes_field = ""

    for item in summary_items:
        lower = item.lower()
        if lower.startswith("main concern:"):
            main_concern = item.split(":", 1)[1].strip()
        elif lower.startswith("general impression:"):
            general_impression = item.split(":", 1)[1].strip()
        elif lower.startswith("notes:"):
            notes_field = item.split(":", 1)[1].strip()

    summary_text_items = [
        item for item in summary_items
        if not item.lower().startswith(("main concern:", "general impression:", "notes:"))
    ]

    topics = infer_meeting_topics(summary_items, next_steps)

    chunk_text = (
        f"Client Meeting Note\n"
        f"Client: {client}\n"
        f"Customer ID: {customer_id}\n"
        f"Date: {date_value}\n"
        f"Industry: {industry}\n"
        f"Country: {country}\n"
        f"Attendees: {', '.join(attendees)}\n\n"
        f"Meeting Summary:\n" +
        "\n".join(f"- {item}" for item in summary_text_items) +
        f"\n\nMain Concern: {main_concern}" +
        f"\nGeneral Impression: {general_impression}" +
        f"\nNotes: {notes_field}" +
        "\n\nNext Steps:\n" +
        "\n".join(f"- {item}" for item in next_steps)
    ).strip()

    metadata = {
        "record_type": "meeting_note",
        "doc_type": "meeting_note",
        "source_file": file_path.name,
        "source_path": str(file_path),
        "client_name": client,
        "customer_id": customer_id,
        "meeting_date": date_value,
        "industry": industry,
        "country": country,
        "attendees": attendees,
        "main_concern": main_concern,
        "general_impression": general_impression,
        "topics": topics
    }

    return [{
        "id": stable_chunk_id(file_path.name, "meeting_note", chunk_text),
        "text": chunk_text,
        "metadata": metadata
    }]


# -----------------------------
# Main dispatcher
# -----------------------------
def process_file(file_path: Path) -> List[Dict[str, Any]]:
    raw_text = file_path.read_text(encoding="utf-8")
    text = clean_text(raw_text)
    file_type = detect_file_type(file_path, text)

    if file_type == "faq":
        return process_faq_file(file_path, text)

    if file_type == "meeting_note":
        return process_meeting_note_file(file_path, text)

    return process_document_file(file_path, text)


def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

    all_chunks = []

    for file_path in sorted(INPUT_DIR.rglob("*.txt")):
        chunks = process_file(file_path)
        all_chunks.extend(chunks)
        print(f"Processed {file_path.name}: {len(chunks)} chunks")

    write_jsonl(all_chunks, OUTPUT_FILE)
    print(f"\nSaved {len(all_chunks)} chunks to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()