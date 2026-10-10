"""Step 5: evaluate three RAG configurations on questions.json.

Metrics per answerable question:
  - retrieval_hit    : at least one required source paper among the chunks passed to the answer step
  - evidence_hit_pre : an evidence snippet appears in the chunks retrieved BEFORE CRAG grading
  - evidence_hit     : an evidence snippet appears in the chunks passed to the answer step
                       (pre=True, hit=False means the CRAG grader dropped the right chunk)
  - coverage         : fraction of key points the answer covers (LLM judge, one yes/no per key point)
  - false_refusal    : the system declined to answer a question that is answerable
Metric per unanswerable question:
  - abstained        : the system declined instead of making up an answer
Also: latency per question.

Each question is run N_RUNS times per config, because the pipeline is not deterministic.

    uv run python evaluate.py
"""
from dotenv import load_dotenv
load_dotenv()

import json
import time
from pathlib import Path
from statistics import mean

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from rag_graph import build_graph
from check_questions import normalize

QUESTIONS_FILE = Path("questions.json")
RESULTS_FILE = Path("results/eval_results.json")
N_RUNS = 2

CONFIGS = {
    "baseline":    dict(multi_query=False, crag=False),
    "multi_query": dict(multi_query=True,  crag=False),
    "mq_crag":     dict(multi_query=True,  crag=True),
}

# A stronger model than the one being tested, to reduce self-preference bias
judge_llm = ChatOpenAI(model="gpt-4o", temperature=0, timeout=60, max_retries=2)


# ---------- 1) Judges ----------
class KeyPointVerdict(BaseModel):
    covered: bool = Field(description="True if the answer states this key point (wording may differ)")


KEY_POINT_PROMPT = ChatPromptTemplate.from_template(
    "You are checking whether an answer covers one specific key point.\n"
    "Judge only this key point. Paraphrases count; vague or contradictory statements do not.\n\n"
    "Question: {question}\n"
    "Key point: {key_point}\n\n"
    "Answer to check:\n{answer}"
)
key_point_judge = KEY_POINT_PROMPT | judge_llm.with_structured_output(KeyPointVerdict)


class AbstainVerdict(BaseModel):
    declined: bool = Field(
        description="True if the answer says it cannot answer or the information is not available, "
                    "False if it gives a substantive answer"
    )


ABSTAIN_PROMPT = ChatPromptTemplate.from_template(
    "Does the following answer decline to answer the question (e.g. says it doesn't know, "
    "or that the information is not in its sources)?\n\n"
    "Question: {question}\n\n"
    "Answer:\n{answer}"
)
abstain_judge = ABSTAIN_PROMPT | judge_llm.with_structured_output(AbstainVerdict)


def coverage(question: str, answer: str, key_points: list[str]) -> float:
    """Fraction of key points covered by the answer. Judges all key points in parallel."""

    inputs = [{"question": question, "key_point": kp, "answer": answer} for kp in key_points]
    verdicts = key_point_judge.batch(inputs)
    return mean(v.covered for v in verdicts)


def declined(question: str, answer: str) -> bool:
    return abstain_judge.invoke({"question": question, "answer": answer}).declined


# ---------- 2) Run one question once ----------
def evidence_found(docs, evidence: list[str]):
    """True if any evidence snippet appears in the text of the given chunks."""
    if not evidence:
        return None
    text = normalize(" ".join(d.page_content for d in docs))
    return any(normalize(snippet) in text for snippet in evidence)


def run_once(graph, q: dict) -> dict:
    start = time.time()
    result = graph.invoke({"question": q["question"]})
    latency = time.time() - start

    answer = result["answer"]
    final_docs = result.get("documents", [])
    # Graphs without CRAG have no grading step, so "before grading" is the same as the final list
    pre_docs = result.get("retrieved_documents", final_docs)
    retrieved_papers = sorted({d.metadata["paper"] for d in final_docs})

    record = {
        "id": q["id"],
        "type": q["type"],
        "dev": q.get("dev", False),
        "answer": answer,
        "retrieved_papers": retrieved_papers,
        "retrieved_chunks": [f"{d.metadata['paper']} p.{d.metadata['page']}" for d in final_docs],
        "latency": round(latency, 2),
    }

    if q["type"] == "unanswerable":
        record["abstained"] = declined(q["question"], answer)
    else:
        record["retrieval_hit"] = any(paper in retrieved_papers for paper in q["required_sources"])
        record["evidence_hit_pre"] = evidence_found(pre_docs, q["evidence"])   # retrieved at all?
        record["evidence_hit"] = evidence_found(final_docs, q["evidence"])     # reached the answer step?
        record["coverage"] = coverage(q["question"], answer, q["key_points"])
        record["false_refusal"] = declined(q["question"], answer)
    return record


# ---------- 3) Summaries ----------

METRICS = ["retrieval_hit", "evidence_hit_pre", "evidence_hit", "coverage", "false_refusal", "abstained", "latency"]


def summarize(records: list[dict]) -> dict:
    if not records:
        return {k: None for k in METRICS}
    answerable = [r for r in records if r["type"] != "unanswerable"]
    unanswerable = [r for r in records if r["type"] == "unanswerable"]

    def avg(rows, key):
        values = [r[key] for r in rows if r.get(key) is not None]
        return round(mean(values), 3) if values else None

    return {
        "retrieval_hit":    avg(answerable, "retrieval_hit"),
        "evidence_hit_pre": avg(answerable, "evidence_hit_pre"),
        "evidence_hit":     avg(answerable, "evidence_hit"),
        "coverage":         avg(answerable, "coverage"),
        "false_refusal":    avg(answerable, "false_refusal"),
        "abstained":        avg(unanswerable, "abstained"),
        "latency":          avg(records, "latency"),
    }


def print_table(title: str, rows: dict[str, dict]):
    print(f"\n{title}")
    print(f"{'config':<14}" + "".join(f"{c:>18}" for c in METRICS))
    for name, s in rows.items():
        print(f"{name:<14}" + "".join(f"{str(s[c]):>18}" for c in METRICS))


# ---------- 4) Main ----------
def main():
    questions = json.loads(QUESTIONS_FILE.read_text())
    #questions = [q for q in questions if q["id"] in ("q03", "q04", "q07")]   # TEMP: smoke test
    all_records = {}

    for name, kwargs in CONFIGS.items():
        graph = build_graph(**kwargs)
        records = []
        for q in questions:
            for run in range(N_RUNS):
                print(f"[{name}] {q['id']} run {run + 1}/{N_RUNS}")
                record = run_once(graph, q)
                record["run"] = run
                records.append(record)
        all_records[name] = records

    RESULTS_FILE.parent.mkdir(exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(all_records, indent=2, ensure_ascii=False))

    # Headline numbers exclude dev questions (we looked at them while building the system)
    print_table("Held-out questions (dev excluded)",
                {n: summarize([r for r in rs if not r["dev"]]) for n, rs in all_records.items()})
    print_table("All questions",
                {n: summarize(rs) for n, rs in all_records.items()})

    for qtype in ["factual", "conceptual", "comparison"]:
        print_table(f"By type: {qtype}",
                    {n: summarize([r for r in rs if r["type"] == qtype]) for n, rs in all_records.items()})


if __name__ == "__main__":
    main()

