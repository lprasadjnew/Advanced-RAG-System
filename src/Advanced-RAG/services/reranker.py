"""
Cross-encoder reranker.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Trained on MS MARCO passage ranking
  - Excellent precision vs speed trade-off
  - Returns relevance logits; higher = more relevant
"""

from typing import List, Tuple

from langchain_core.documents import Document
from sentence_transformers.cross_encoder import CrossEncoder

from config import settings

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_cross_encoder: CrossEncoder | None = None


def _get_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(_MODEL_NAME)
    return _cross_encoder


def rerank(
    query: str,
    doc_score_pairs: List[Tuple[Document, float]],
    top_k: int = None,
) -> List[Document]:
    """
    Rerank retrieved documents using a cross-encoder.

    Args:
        query          : User's question.
        doc_score_pairs: Output of vector_store.retrieve_documents().
        top_k          : Number of top documents to return.

    Returns:
        Reranked list of Documents (most relevant first).
    """
    top_k = top_k or settings.TOP_K_RERANKED
    if not doc_score_pairs:
        return []

    encoder = _get_encoder()
    docs = [d for d, _ in doc_score_pairs]
    pairs = [(query, doc.page_content) for doc in docs]

    scores = encoder.predict(pairs)

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]
