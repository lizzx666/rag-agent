"""Summarize tokens, cost and latency per pipeline config from the LangSmith traces of profile_configs.py.

Wait a minute or two after profile_configs.py finishes: LangSmith fills in token counts and costs
asynchronously.

    uv run python cost_report.py
"""
from dotenv import load_dotenv
load_dotenv()

from collections import defaultdict
from statistics import mean, median

from langsmith import Client

PROJECT = "rag-agent-profile"
RUN_TAG = "profile-v1"


def main():
    client = Client()
    # Root runs only: one per (config, question). Their token and cost fields already
    # include every LLM call made inside the graph.
    runs = [r for r in client.list_runs(project_name=PROJECT, is_root=True) if RUN_TAG in (r.tags or [])]
    print(f"Found {len(runs)} root runs with tag {RUN_TAG}")

    by_config = defaultdict(list)
    for run in runs:
        metadata = (run.extra or {}).get("metadata", {})
        by_config[metadata["config"]].append({
            "latency": (run.end_time - run.start_time).total_seconds(),
            "tokens": run.total_tokens or 0,
            "cost": float(run.total_cost or 0),
        })

    print(f"\n{'config':<14}{'runs':>6}{'tokens/q':>10}{'cost/q ($)':>12}{'p50 latency':>13}{'mean latency':>14}")
    for name, rows in by_config.items():
        n_runs = len(rows)
        mean_tokens = mean(row["tokens"] for row in rows)
        mean_cost = mean(row["cost"] for row in rows)
        median_latency = median(row["latency"] for row in rows)
        mean_latency = mean(row["latency"] for row in rows)

        print(f"{name:<14}{n_runs:>6}{mean_tokens:>10.1f}{mean_cost:>12.6f}{median_latency:>13.2f}{mean_latency:>14.2f}")


if __name__ == "__main__":
    main()