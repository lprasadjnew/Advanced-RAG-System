"""
Core RAG orchestration layer.

Query pipeline:
  1. Retrieve top-K documents from Qdrant (dense vector search).
  2. Rerank with cross-encoder → keep top-N.
  3. Build context-aware prompt including conversation history.
  4. Generate answer with Gemini.
  5. Persist Q&A to SQLite conversation memory.

Ingestion pipeline:
  1. Extract preview text for domain validation.
  2. Validate domain with Gemini classifier.
  3. If valid: chunk → embed → upsert to Qdrant.
"""

from typing import List

from google import genai

import database as db
from config import settings
from services.document_processor import compute_content_hash, extract_preview_text, load_and_chunk
from services.domain_validator import validate_domain
from services.reranker import rerank
from services.vector_store import is_duplicate, retrieve_documents, store_documents

_client = genai.Client(api_key=settings.GOOGLE_API_KEY)


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a knowledgeable assistant specialised in {domain_description}.
Answer the user's question using ONLY the provided context passages.
If the context does not contain enough information to answer, say so honestly.
Be concise, accurate, and helpful. Format your response clearly.
"""

_RAG_PROMPT = """\
{system}

--- Retrieved Context ---
{context}
--- End of Context ---

--- Conversation History ---
{history}
--- End of History ---

User: {question}
Assistant:\
"""


# ── Ingestion ─────────────────────────────────────────────────────────────────

def ingest_document(file_path: str) -> tuple[bool, str]:
    """
    Full ingestion pipeline.

    Order of checks (fast → slow, cheap → expensive):
      1. Extract text  — fail fast if file is unreadable.
      2. Duplicate check against Qdrant  — no API call, pure DB lookup.
      3. Domain validation via Gemini  — API call only when content is new.
      4. Chunk → embed → store.

    Returns:
        (success: bool, message: str)
    """
    # Step 1 — extract full text & compute content fingerprint
    preview = extract_preview_text(file_path)
    if not preview.strip():
        return False, "Could not extract text from the uploaded file."

    content_hash = compute_content_hash(file_path)

    # Step 2 — duplicate check (content-based, filename-agnostic)
    duplicate, original_source = is_duplicate(content_hash)
    if duplicate:
        return (
            False,
            f"This document's content has already been processed and stored "
            f"in the knowledge base (originally uploaded as '{original_source}'). "
            f"No changes were made.",
        )

    # Step 3 — domain validation (Gemini API call)
    is_valid, reason = validate_domain(preview)
    if not is_valid:
        return (
            False,
            f"Uploaded document is not related to the configured domain "
            f"({settings.DOMAIN_DESCRIPTION}). Reason: {reason}",
        )

    # Step 4 — chunk, embed & store (hash stamped onto every chunk)
    chunks = load_and_chunk(file_path)
    if not chunks:
        return False, "Document appears to be empty or could not be parsed."

    count = store_documents(chunks, content_hash)

    return (
        True,
        f"The document has been successfully uploaded and processed into "
        f"{count} chunks using sophisticated chunking strategy.",
    )


# ── Query ─────────────────────────────────────────────────────────────────────

def _build_history_text(messages: List[dict]) -> str:
    if not messages:
        return "No prior conversation."
    lines = []
    for m in messages:
        role = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines)


def _build_context_text(docs) -> str:
    if not docs:
        return "No relevant context found."
    parts = []
    for i, doc in enumerate(docs, start=1):
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "")
        ref = f"[{i}] ({src}" + (f", p.{page}" if page else "") + ")"
        parts.append(f"{ref}\n{doc.page_content}")
    return "\n\n".join(parts)


def answer_query(question: str, session_id: str) -> str:
    """
    Full RAG query pipeline.

    Args:
        question  : User's question.
        session_id: Current conversation session.

    Returns:
        Assistant's answer string.
    """
    # Retrieve + rerank
    retrieved = retrieve_documents(question, k=settings.TOP_K_RETRIEVAL)
    top_docs = rerank(question, retrieved, top_k=settings.TOP_K_RERANKED)

    # Load conversation history
    history = db.get_conversation(session_id)

    # Build prompt
    system = _SYSTEM_PROMPT.format(domain_description=settings.DOMAIN_DESCRIPTION)
    context = _build_context_text(top_docs)
    history_text = _build_history_text(history)

    prompt = _RAG_PROMPT.format(
        system=system,
        context=context,
        history=history_text,
        question=question,
    )

    # Generate
    response = _client.models.generate_content(
        model=settings.GEMINI_MODEL,
        contents=prompt,
    )
    answer = (response.text or "").strip()

    # Persist to memory
    db.save_message(session_id, "user", question)
    db.save_message(session_id, "assistant", answer)

    return answer
