"""Print retrieved chunks and the answer for chosen questions, to check whether answers are grounded."""
import json
from pathlib import Path

RESULTS_FILE = Path("results/eval_results.json")
CONFIG = "baseline"
IDS = ["q05", "q09"]
RUN = 0

records = json.loads(RESULTS_FILE.read_text())[CONFIG]
for r in records:
    if r["id"] in IDS and r["run"] == RUN:
        print("=" * 80)
        print(f"{r['id']}  coverage={r['coverage']}  evidence_hit={r['evidence_hit']}")
        print("Retrieved chunks:", r["retrieved_chunks"])
        print("\nAnswer:\n" + r["answer"])