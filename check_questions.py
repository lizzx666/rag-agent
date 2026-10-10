"""Sanity-check questions.json against the PDFs before running the evaluation.

Checks:
  1. every required/optional source is a real paper name in data/
  2. every evidence snippet really appears in one of the cited papers
  3. for unanswerable questions, none of the absent_keywords appears in ANY paper

No API calls; safe to run offline:
    uv run python check_questions.py
"""
import json
import re
from pathlib import Path

from pypdf import PdfReader

DATA_DIR = Path("data")
QUESTIONS_FILE = Path("questions.json")

LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl", "’": "'", "‘": "'", "“": '"', "”": '"'}


def normalize(text: str) -> str:
    """Make PDF text and evidence comparable: fix ligatures, join words split
    across lines ('un- der' -> 'under'), collapse whitespace, lowercase."""
    for bad, good in LIGATURES.items():
        text = text.replace(bad, good)
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def load_papers() -> dict[str, str]:
    papers = {}
    for pdf_path in sorted(DATA_DIR.glob("*.pdf")):
        reader = PdfReader(str(pdf_path))
        papers[pdf_path.stem] = normalize(" ".join(page.extract_text() or "" for page in reader.pages))
    return papers


def main():
    papers = load_papers()
    questions = json.loads(QUESTIONS_FILE.read_text())
    problems = 0

    for q in questions:
        cited = q.get("required_sources", []) + q.get("optional_sources", [])

        # 1) sources must be real paper names
        for src in cited:
            if src not in papers:
                print(f"[{q['id']}] unknown source '{src}' (known: {sorted(papers)})")
                problems += 1

        # 2) evidence must appear in one of the cited papers
        for snippet in q.get("evidence", []):
            if not any(normalize(snippet) in papers.get(src, "") for src in cited):
                print(f"[{q['id']}] evidence not found in {cited}: \"{snippet[:60]}...\"")
                problems += 1

        # 3) unanswerable: keywords must be absent from the whole corpus
        if q["type"] == "unanswerable":
            for kw in q.get("absent_keywords", []):
                found_in = [name for name, text in papers.items() if normalize(kw) in text]
                if found_in:
                    print(f"[{q['id']}] '{kw}' appears in {found_in}: the question may be answerable")
                    problems += 1

    print(f"\nChecked {len(questions)} questions, {problems} problem(s) found.")


if __name__ == "__main__":
    main()
