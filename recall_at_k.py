"""Diagnostic: Recall@k of the baseline retriever, measured with evidence snippets.

For each answerable question:
  1. search the vector store with the raw question, top MAX_K chunks
  2. find the rank of the first chunk that contains any evidence snippet
Then Recall@k = fraction of questions whose evidence rank <= k.

If Recall@k rises a lot from k=5 to k=20, the right chunks are retrieved but ranked too low
-> "retrieve more, then rerank" is worth trying.
If it stays flat, the retriever never finds them -> a different retrieval method is needed (e.g. BM25 hybrid).

Only embedding calls, no LLM calls:
    uv run python recall_at_k.py
"""
from dotenv import load_dotenv
load_dotenv()

import json
from pathlib import Path

from langchain_core.documents import Document

from index import get_vectorstore
from check_questions import normalize

QUESTIONS_FILE = Path("questions.json")
RESULTS_FILE = Path("results/recall_at_k.json")
MAX_K = 30
K_VALUES = [1, 3, 5, 10, 20, 30]


def evidence_rank(docs: list[Document], evidence: list[str]) -> int | None:
    """1-based rank of the first chunk that contains any evidence snippet, or None if not found."""
    targets = [normalize(s) for s in evidence]
    for i, doc in enumerate(docs, start=1):
        text = normalize(doc.page_content)
        if any(t in text for t in targets):
            return i
    return None


def recall_at(ranks: list[int | None], k: int) -> float:
    """Fraction of questions whose evidence rank is not None and <= k."""
    count = sum(1 for r in ranks if r is not None and r <= k)
    return round(count / len(ranks), 3) if ranks else 0.0


def main():
    questions = json.loads(QUESTIONS_FILE.read_text())
    answerable = [q for q in questions if q["type"] != "unanswerable" and q["evidence"]]

    vectorstore = get_vectorstore()
    # One search per question (sequential); same embedding model and store as the RAG pipeline
    all_results = [vectorstore.similarity_search(q["question"], k=MAX_K) for q in answerable]

    per_question = []
    for q, docs in zip(answerable, all_results):
        rank = evidence_rank(docs, q["evidence"])
        per_question.append({"id": q["id"], "type": q["type"], "dev": q.get("dev", False), "rank": rank})

    # Per-question ranks
    print(f"\n{'id':<6}{'type':<13}{'dev':<7}{'evidence rank':>14}")
    for r in per_question:
        rank_str = str(r["rank"]) if r["rank"] else f">{MAX_K}"
        print(f"{r['id']:<6}{r['type']:<13}{str(r['dev']):<7}{rank_str:>14}")

    # Recall@k curve: all answerable questions, and held-out only
    curves = {}
    for name, rows in [("all", per_question), ("held_out", [r for r in per_question if not r["dev"]])]:
        ranks = [r["rank"] for r in rows]
        curves[name] = {f"recall@{k}": recall_at(ranks, k) for k in K_VALUES}

    print(f"\n{'subset':<10}" + "".join(f"{'@' + str(k):>8}" for k in K_VALUES))
    for name, curve in curves.items():
        print(f"{name:<10}" + "".join(f"{curve[f'recall@{k}']:>8}" for k in K_VALUES))

    RESULTS_FILE.parent.mkdir(exist_ok=True)
    RESULTS_FILE.write_text(json.dumps({"per_question": per_question, "curves": curves}, indent=2))


if __name__ == "__main__":
    main()