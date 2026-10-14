"""Mean coverage per question and config, from the latest evaluation run."""
import json
from pathlib import Path
from statistics import mean

RESULTS_FILE = Path("results/eval_results.json")

records = json.loads(RESULTS_FILE.read_text())
configs = list(records)
first = records[configs[0]]
questions = sorted({(r["id"], r["type"], r["dev"]) for r in first if r["type"] != "unanswerable"})

print(f"{'id':<6}{'type':<13}{'dev':<7}" + "".join(f"{c:>14}" for c in configs))
for qid, qtype, dev in questions:
    row = []
    for c in configs:
        values = [r["coverage"] for r in records[c] if r["id"] == qid]
        row.append(f"{mean(values):.2f}")
    print(f"{qid:<6}{qtype:<13}{str(dev):<7}" + "".join(f"{v:>14}" for v in row))