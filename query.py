import re
import chromadb
import ollama
from sentence_transformers import SentenceTransformer

# --- Config ---
CHROMA_PATH = "chroma_store"
COLLECTION_NAME = "my_docs"
LLM_MODEL = "llama3"
TOP_K = 12

# --- Load embedding model ---
embedder = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)

# --- Connect to Chroma ---
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME)


TYPE_MAP = {
    "[ticket]": "support_ticket",
    "[lead]": "lead",
    "[customer]": "customer",
    "[sales_note]": "sales_note",
}


def normalize_customer_id(text: str) -> str:
    m = re.search(r"(\d+)", text or "")
    if not m:
        return ""
    return m.group(1).zfill(3)


def extract_customer_id(query: str) -> str | None:
    """
    Supports:
    - customer 027
    - customer_id 027
    - cust 027
    - cust_027
    - cust-027
    """
    patterns = [
        r"\bcustomer_id\s*[:=]?\s*0*(\d+)\b",
        r"\bcustomer\s*[:#_\-\s]?\s*0*(\d+)\b",
        r"\bcust\s*[:#_\-\s]?\s*0*(\d+)\b",
    ]
    for pattern in patterns:
        m = re.search(pattern, query, flags=re.IGNORECASE)
        if m:
            return m.group(1).zfill(3)
    return None


def build_where_filter(record_type_hint: str | None, customer_id_norm: str | None):
    clauses = []

    if record_type_hint:
        clauses.append({"record_type": record_type_hint})

    if customer_id_norm:
        clauses.append({"customer_id_norm": customer_id_norm})
        clauses.append({"entity_scope": "single_customer"})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def is_customer_query(record_type_hint: str | None, query: str) -> bool:
    if record_type_hint == "customer":
        return True
    return extract_customer_id(query) is not None


def dedupe_and_prioritize(chunks, metadatas, customer_id_norm: str | None):
    seen = set()
    ranked = []

    for chunk, meta in zip(chunks, metadatas):
        key = chunk[:250]
        if key in seen:
            continue
        seen.add(key)

        score = 0
        meta_customer = str(meta.get("customer_id_norm", "")).strip()
        chunk_type = str(meta.get("chunk_type", "")).strip()

        if customer_id_norm and meta_customer == customer_id_norm:
            score += 100
        if chunk_type == "row":
            score += 20
        elif chunk_type == "row_group":
            score += 10
        elif chunk_type in {"thread_summary", "individual_message"}:
            score += 15

        ranked.append((score, chunk, meta))

    ranked.sort(key=lambda x: x[0], reverse=True)

    deduped_chunks = []
    deduped_sources = []
    deduped_metas = []

    for _, chunk, meta in ranked:
        deduped_chunks.append(chunk)
        source = (
            meta.get("source")
            or meta.get("source_file")
            or meta.get("source_path")
            or "unknown"
        )
        deduped_sources.append(source)
        deduped_metas.append(meta)

    return deduped_chunks, deduped_sources, deduped_metas


def retrieve(query: str, record_type_hint: str | None = None):
    customer_id_norm = extract_customer_id(query)

    if is_customer_query(record_type_hint, query) and not customer_id_norm:
        return {
            "error": "Customer query detected but no customer_id was found in the question.",
            "chunks": [],
            "sources": [],
            "metadatas": [],
            "customer_id_norm": None,
        }

    query_embedding = embedder.encode(
        [f"search_query: {query}"],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).tolist()

    where = build_where_filter(record_type_hint, customer_id_norm)

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=TOP_K,
        include=["documents", "metadatas"],
        where=where,
    )

    chunks = results["documents"][0] if results.get("documents") else []
    metadatas = results["metadatas"][0] if results.get("metadatas") else []

    chunks, sources, metadatas = dedupe_and_prioritize(chunks, metadatas, customer_id_norm)

    return {
        "error": None,
        "chunks": chunks,
        "sources": sources,
        "metadatas": metadatas,
        "customer_id_norm": customer_id_norm,
    }


def ask(query: str, record_type_hint: str | None = None) -> dict:
    retrieval = retrieve(query, record_type_hint=record_type_hint)

    if retrieval["error"]:
        return {
            "answer": retrieval["error"],
            "sources": [],
        }

    chunks = retrieval["chunks"]
    sources = retrieval["sources"]
    customer_id_norm = retrieval["customer_id_norm"]

    if not chunks:
        return {
            "answer": "I don't have enough information to answer that.",
            "sources": [],
        }

    context = "\n\n---\n\n".join(chunks)

    customer_guard = ""
    if customer_id_norm:
        customer_guard = f"""
CRITICAL ENTITY RULES:
- The user is asking about customer {customer_id_norm}.
- Use ONLY context that belongs to customer {customer_id_norm}.
- If any context appears inconsistent with customer {customer_id_norm}, ignore it.
- Never mix data from different customers.
- If the answer for customer {customer_id_norm} is incomplete in the context, say so explicitly.
"""

    prompt = f"""You are a helpful assistant with access to structured business data (support tickets, CRM records, sales notes).

Answer the question using ONLY the context below.
If the data is tabular, reason carefully across all relevant rows provided.
If the answer requires data not shown in the context, say so explicitly rather than guessing.
If the answer is not in the context at all, say "I don't have enough information to answer that."
{customer_guard}

Context:
{context}

Question: {query}

Answer:"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    answer = response["message"]["content"]
    unique_sources = sorted(set(sources))

    return {"answer": answer, "sources": unique_sources}


if __name__ == "__main__":
    print("RAG Query Interface")
    print("  Tip: prefix your question with a record type to narrow results.")
    print("  e.g. [customer] What is the latest update for customer 027?")
    print("  Supported prefixes: [ticket] [lead] [customer] [sales_note]")
    print("  Customer queries must include a customer_id.")
    print("  Type 'quit' to exit.\n")

    while True:
        raw = input("Ask a question (or 'quit'): ").strip()
        if raw.lower() in ("quit", "exit", "q"):
            break
        if not raw:
            continue

        record_type_hint = None
        query = raw
        for prefix, rtype in TYPE_MAP.items():
            if raw.lower().startswith(prefix):
                record_type_hint = rtype
                query = raw[len(prefix):].strip()
                break

        result = ask(query, record_type_hint=record_type_hint)
        print(f"\nAnswer: {result['answer']}")
        print(f"Sources: {', '.join(result['sources']) if result['sources'] else 'None'}\n")
