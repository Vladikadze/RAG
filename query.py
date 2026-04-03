import chromadb
import ollama
from sentence_transformers import SentenceTransformer

# --- Config ---
CHROMA_PATH = "chroma_store"
COLLECTION_NAME = "my_docs"
LLM_MODEL = "llama3"
TOP_K = 10

# --- Load embedding model ---
embedder = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)

# --- Connect to Chroma ---
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME)


def retrieve(query: str, record_type_hint: str | None = None):
    # Add task prefix required by nomic-embed-text-v1
    query_embedding = embedder.encode(
        [f"search_query: {query}"],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).tolist()

    # Optional metadata filter (e.g. narrow to a specific record type)
    where = {"record_type": record_type_hint} if record_type_hint else None

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=TOP_K,
        include=["documents", "metadatas"],
        where=where,
    )

    chunks = results["documents"][0] if results.get("documents") else []
    metadatas = results["metadatas"][0] if results.get("metadatas") else []

    # Deduplicate by content fingerprint to avoid row/row_group overlap
    seen = set()
    deduped_chunks, deduped_sources = [], []
    for chunk, meta in zip(chunks, metadatas):
        key = chunk[:200]
        if key not in seen:
            seen.add(key)
            deduped_chunks.append(chunk)
            source = (
                meta.get("source")
                or meta.get("source_file")
                or meta.get("source_path")
                or "unknown"
            )
            deduped_sources.append(source)

    return deduped_chunks, deduped_sources


def ask(query: str, record_type_hint: str | None = None) -> dict:
    chunks, sources = retrieve(query, record_type_hint=record_type_hint)

    if not chunks:
        return {
            "answer": "I don't have enough information to answer that.",
            "sources": [],
        }

    context = "\n\n---\n\n".join(chunks)

    prompt = f"""You are a helpful assistant with access to structured business data (support tickets, CRM records, sales notes).
Answer the question using ONLY the context below.
If the data is tabular, reason carefully across all rows provided.
If the answer requires data not shown in the context, say so explicitly rather than guessing.
If the answer is not in the context at all, say "I don't have enough information to answer that."

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
    print("  e.g.  [ticket] How many open tickets does Acme have?")
    print("  Supported prefixes: [ticket] [lead] [customer] [sales_note]")
    print("  Type 'quit' to exit.\n")

    TYPE_MAP = {
        "[ticket]": "support_ticket",
        "[lead]": "lead",
        "[customer]": "customer",
        "[sales_note]": "sales_note",
    }

    while True:
        raw = input("Ask a question (or 'quit'): ").strip()
        if raw.lower() in ("quit", "exit", "q"):
            break
        if not raw:
            continue

        # Parse optional record type prefix
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