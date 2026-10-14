"""Cross-encoder reranker, shared by the RAG graph and the Recall@k diagnostic.

A cross-encoder reads the query and one chunk together and outputs a relevance score.
More accurate than embedding similarity, but it must run once per (query, chunk) pair,
so we only apply it to a small candidate set from the vector search.
"""
from dotenv import load_dotenv
load_dotenv()   # must run before importing sentence_transformers: HF_ENDPOINT is read at import time

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_model = None   # loaded on first use, then reused (loading takes a few seconds)


def get_reranker() -> CrossEncoder:
    global _model
    if _model is None:
        print(f"Loading reranker {RERANK_MODEL}")
        _model = CrossEncoder(RERANK_MODEL)
    return _model


def rerank(query: str, docs: list[Document], top_n: int | None = None) -> list[Document]:
    """Score every (query, chunk) pair with the cross-encoder and return chunks sorted by score.
    Returns all chunks if top_n is None, otherwise the best top_n."""
    if not docs:
        return []

    pairs = [(query, doc.page_content) for doc in docs]
    scores = get_reranker().predict(pairs)
    sorted_docs = [doc for _, doc in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]
    if top_n is not None:
        sorted_docs = sorted_docs[:top_n]
    return sorted_docs


if __name__ == "__main__":
    # Quick check: the Q-BLEU chunk should move up after reranking

    from index import get_vectorstore

    query = "What is Q-BLEU?"
    candidates = get_vectorstore().similarity_search(query, k=20)
    print("\nBefore rerank:")
    for i, d in enumerate(candidates[:5], start=1):
        print(f"  {i}. {d.metadata['paper']} p.{d.metadata['page']}: {d.page_content[:80]!r}")
    print("\nAfter rerank:")
    for i, d in enumerate(rerank(query, candidates, top_n=5), start=1):
        print(f"  {i}. {d.metadata['paper']} p.{d.metadata['page']}: {d.page_content[:80]!r}")