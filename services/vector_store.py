"""Qdrant vector store operations."""

from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, VectorParams

from config import settings

# ── Clients ───────────────────────────────────────────────────────────────────

_embeddings = GoogleGenerativeAIEmbeddings(
    model=settings.GEMINI_EMBEDDING_MODEL,
    google_api_key=settings.GOOGLE_API_KEY,
    task_type="retrieval_document",
)

_query_embeddings = GoogleGenerativeAIEmbeddings(
    model=settings.GEMINI_EMBEDDING_MODEL,
    google_api_key=settings.GOOGLE_API_KEY,
    task_type="retrieval_query",
)

_qdrant_client = QdrantClient(
    host=settings.QDRANT_HOST,
    port=settings.QDRANT_PORT,
)

VECTOR_SIZE = 3072  # gemini-embedding-001 output dimension


# ── Collection bootstrap ───────────────────────────────────────────────────────

def ensure_collection() -> None:
    """Create the Qdrant collection if it does not already exist."""
    existing = [c.name for c in _qdrant_client.get_collections().collections]
    if settings.QDRANT_COLLECTION not in existing:
        _qdrant_client.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


# ── Duplicate detection ───────────────────────────────────────────────────────

def is_duplicate(content_hash: str) -> tuple[bool, Optional[str]]:
    """
    Check whether a document with this content hash already exists in Qdrant.

    Searches the payload field  metadata.content_hash  which is stamped on
    every chunk at ingestion time.

    Returns:
        (True, original_source_name)  if duplicate found.
        (False, None)                 if content is new.
    """
    ensure_collection()
    points, _ = _qdrant_client.scroll(
        collection_name=settings.QDRANT_COLLECTION,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="metadata.content_hash",
                    match=MatchValue(value=content_hash),
                )
            ]
        ),
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    if points:
        original = (
            points[0].payload.get("metadata", {}).get("source", "unknown file")
        )
        return True, original
    return False, None


# ── Write ─────────────────────────────────────────────────────────────────────

def store_documents(chunks: List[Document], content_hash: str) -> int:
    """
    Embed and upsert chunks into Qdrant.
    Stamps content_hash into every chunk's metadata for future dedup checks.
    Returns the number of chunks stored.
    """
    ensure_collection()
    for chunk in chunks:
        chunk.metadata["content_hash"] = content_hash
    store = QdrantVectorStore(
        client=_qdrant_client,
        collection_name=settings.QDRANT_COLLECTION,
        embedding=_embeddings,
    )
    store.add_documents(chunks)
    return len(chunks)


# ── Read ──────────────────────────────────────────────────────────────────────

def retrieve_documents(query: str, k: int = None) -> List[Tuple[Document, float]]:
    """
    Retrieve top-k documents with relevance scores.
    Uses retrieval_query task type for query embedding.
    """
    k = k or settings.TOP_K_RETRIEVAL
    ensure_collection()
    store = QdrantVectorStore(
        client=_qdrant_client,
        collection_name=settings.QDRANT_COLLECTION,
        embedding=_query_embeddings,
    )
    results = store.similarity_search_with_relevance_scores(query, k=k)
    return results
