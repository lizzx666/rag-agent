"""Run every pipeline config once on every question, tagged for LangSmith, to measure cost and latency.

No judging here (judge calls would mix into the traces); quality numbers come from evaluate.py.
Traces go to a separate LangSmith project so they don't mix with ad-hoc runs.

    LANGSMITH_PROJECT=rag-agent-profile uv run python profile_configs.py
"""
from dotenv import load_dotenv
load_dotenv()

import json
import logging
from pathlib import Path

from rag_graph import build_graph
from evaluate import CONFIGS

logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpx2").setLevel(logging.WARNING)
logger = logging.getLogger("profile")

QUESTIONS_FILE = Path("questions.json")
RUN_TAG = "profile-v1"   # identifies this batch of runs in LangSmith


def main():
    questions = json.loads(QUESTIONS_FILE.read_text())

    for name, kwargs in CONFIGS.items():
        graph = build_graph(**kwargs)
        for q in questions:
            logger.info(f"[{name}] {q['id']}")
            
            config = {
                "run_name": f"{name}:{q['id']}",
                "tags": [RUN_TAG, name],
                "metadata": {"config": name, "question_id": q["id"], "type": q["type"]},
            }
            graph.invoke({"question": q["question"]}, config=config)
            
            


if __name__ == "__main__":
    main()