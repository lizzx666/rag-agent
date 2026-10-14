"""Diagnostic: Recall@k of the retriever, before and after cross-encoder reranking.

For each answerable question:
  1. search the vector store with the raw question, top MAX_K chunks
  2. rank of the first chunk containing an evidence snippet  -> "before"
  3. rerank the top CANDIDATES chunks with the cross-encoder, find the rank again -> "after"
Recall@k = fraction of questions whose evidence rank <= k.

Reranking only reorders the CANDIDATES chunks, so recall@CANDIDATES is the same before and after;
the question is how much it moves evidence from ranks 6-20 into the top 5.

    uv run python recall_at_k.py
"""
from dotenv import load_dotenv
load_dotenv()

import json
from pathlib import Path

from langchain_core.documents import Document

from index import get_vectorstore
from check_questions import normalize
from rerank import rerank

QUESTIONS_FILE = Path("questions.json")
RESULTS_FILE = Path("results/recall_at_k.json")
MAX_K = 30
CANDIDATES = 20                 # fixed in advance, not tuned on these questions
K_BEFORE = [1, 3, 5, 10, 20, 30]
K_AFTER = [1, 3, 5, 10, 20]     # after reranking there are only CANDIDATES chunks


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


def fmt_rank(rank: int | None, limit: int) -> str:
    return str(rank) if rank else f">{limit}"


def main():
    questions = json.loads(QUESTIONS_FILE.read_text())
    answerable = [q for q in questions if q["type"] != "unanswerable" and q["evidence"]]
    vectorstore = get_vectorstore()

    per_question = []
    for q in answerable:
        docs = vectorstore.similarity_search(q["question"], k=MAX_K)
        reranked = rerank(q["question"], docs[:CANDIDATES])
        per_question.append({
            "id": q["id"],
            "type": q["type"],
            "dev": q.get("dev", False),
            "rank_before": evidence_rank(docs, q["evidence"]),
            "rank_after": evidence_rank(reranked, q["evidence"]),
        })

    # Per-question ranks
    print(f"\n{'id':<6}{'type':<13}{'dev':<7}{'before':>8}{'after':>8}")
    for r in per_question:
        print(f"{r['id']:<6}{r['type']:<13}{str(r['dev']):<7}"
              f"{fmt_rank(r['rank_before'], MAX_K):>8}{fmt_rank(r['rank_after'], CANDIDATES):>8}")

    # Recall@k curves, before and after reranking
    curves = {}
    subsets = [("all", per_question), ("held_out", [r for r in per_question if not r["dev"]])]
    for name, rows in subsets:
        curves[f"{name}_before"] = {f"recall@{k}": recall_at([r["rank_before"] for r in rows], k) for k in K_BEFORE}
        curves[f"{name}_after"] = {f"recall@{k}": recall_at([r["rank_after"] for r in rows], k) for k in K_AFTER}

    print(f"\n{'subset':<18}" + "".join(f"{'@' + str(k):>8}" for k in K_BEFORE))
    for name, curve in curves.items():
        print(f"{name:<18}" + "".join(f"{curve.get(f'recall@{k}', '-'):>8}" for k in K_BEFORE))

    RESULTS_FILE.parent.mkdir(exist_ok=True)
    RESULTS_FILE.write_text(json.dumps({"per_question": per_question, "curves": curves}, indent=2))


if __name__ == "__main__":
    main()