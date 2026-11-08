import json
from datetime import datetime
from pathlib import Path

LOG_FILE = Path("logs/queries.jsonl")


def log_query(question, answer, contexts, latency, error=None):
    LOG_FILE.parent.mkdir(exist_ok=True)
    entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "latency_s": round(latency, 2),
        "error": error,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def get_stats():
    if not LOG_FILE.exists():
        return None
    with open(LOG_FILE, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    if not entries:
        return None
    latencies = [e["latency_s"] for e in entries]
    return {
        "total": len(entries),
        "avg_latency": round(sum(latencies) / len(latencies), 2),
        "errors": sum(1 for e in entries if e["error"]),
    }