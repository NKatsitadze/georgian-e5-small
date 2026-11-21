"""Measure CPU speed and memory of embedding models on real Georgian text.

    uv run python speed.py                        # every model in MODELS
    uv run python speed.py --models intfloat/multilingual-e5-small

Each model runs in its own subprocess so memory numbers don't mix. Texts are documents and
queries from GeorgianFAQ. Writes results/speed.md and results/speed.json.
"""

import argparse
import json
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"
MODELS = (
    "intfloat/multilingual-e5-small",
    "intfloat/multilingual-e5-base",
    "intfloat/multilingual-e5-large",
    "BAAI/bge-m3",
    "Snowflake/snowflake-arctic-embed-l-v2.0",
)
MAX_TOKENS = 512  # same cap for every model; bge-m3 would otherwise accept up to 8192
DOCS, QUERIES, LATENCY_RUNS, BATCH_SIZE = 512, 256, 50, 32


def georgian_texts() -> tuple[list[str], list[str]]:
    import mteb

    task = mteb.get_task("GeorgianFAQRetrieval", languages=["kat"])
    task.load_data()
    split = task.dataset[next(iter(task.dataset))]["test"]
    docs = [row["text"] for row in split["corpus"]][:DOCS]
    queries = [row["text"] for row in split["queries"]][:QUERIES]
    return docs, queries


def measure(model_name: str) -> dict:
    """Runs inside the subprocess: load, warm up, then time batches and single queries."""
    import torch
    from sentence_transformers import SentenceTransformer

    docs, queries = georgian_texts()
    model = SentenceTransformer(model_name, device="cpu")
    model.max_seq_length = MAX_TOKENS
    params = sum(p.numel() for p in model.parameters())
    tokens = [len(model.tokenizer(text, truncation=True, max_length=MAX_TOKENS)["input_ids"]) for text in docs]

    model.encode(docs[:BATCH_SIZE], batch_size=BATCH_SIZE)  # warm-up: first calls are always slower

    started = time.perf_counter()
    vectors = model.encode(docs, batch_size=BATCH_SIZE)
    docs_per_second = len(docs) / (time.perf_counter() - started)

    started = time.perf_counter()
    model.encode(queries, batch_size=BATCH_SIZE)
    queries_per_second = len(queries) / (time.perf_counter() - started)

    latencies = []
    for query in queries[:LATENCY_RUNS]:
        started = time.perf_counter()
        model.encode([query], batch_size=1)
        latencies.append((time.perf_counter() - started) * 1000)
    latencies.sort()

    return {
        "model": model_name,
        "params_millions": round(params / 1e6),
        "dimensions": int(vectors.shape[1]),
        "docs_per_second": round(docs_per_second, 1),
        "queries_per_second": round(queries_per_second, 1),
        "query_latency_ms_p50": round(statistics.median(latencies), 1),
        "query_latency_ms_p95": round(latencies[int(len(latencies) * 0.95) - 1], 1),
        "peak_memory_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20),  # bytes on macOS
        "median_doc_tokens": int(statistics.median(tokens)),
        "torch_threads": torch.get_num_threads(),
    }


def run_isolated(model_name: str) -> dict | None:
    print(f"▶ {model_name}", flush=True)
    result = subprocess.run(
        [sys.executable, __file__, "--measure-one", model_name], capture_output=True, text=True, timeout=1800
    )
    lines = [line for line in result.stdout.splitlines() if line.startswith("{")]
    if result.returncode != 0 or not lines:
        print(f"❌ {model_name}: exit {result.returncode}\n{result.stderr[-800:]}", flush=True)
        return None
    row = json.loads(lines[-1])
    print(f"✅ {model_name}: {row['docs_per_second']} docs/s, p50 {row['query_latency_ms_p50']} ms", flush=True)
    return row


def to_markdown(rows: list[dict]) -> str:
    fastest = max(row["docs_per_second"] for row in rows)
    lines = [
        f"CPU only, {rows[0]['torch_threads']} threads, texts capped at {MAX_TOKENS} tokens. "
        f"{DOCS} GeorgianFAQ documents (median {rows[0]['median_doc_tokens']} tokens), batch {BATCH_SIZE}.",
        "",
        "| model | params | dims | docs/s | × slower than fastest | query p50 ms | query p95 ms | peak RAM MB |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in sorted(rows, key=lambda r: -r["docs_per_second"]):
        lines.append(
            f"| {row['model']} | {row['params_millions']}M | {row['dimensions']} | {row['docs_per_second']} "
            f"| {fastest / row['docs_per_second']:.1f}× | {row['query_latency_ms_p50']} "
            f"| {row['query_latency_ms_p95']} | {row['peak_memory_mb']} |"
        )
    return "\n".join(lines) + "\n"


def main(model_names: list[str]) -> int:
    RESULTS_DIR.mkdir(exist_ok=True)
    rows = [row for name in model_names if (row := run_isolated(name))]
    if not rows:
        return 1
    (RESULTS_DIR / "speed.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    table = to_markdown(rows)
    (RESULTS_DIR / "speed.md").write_text(table, encoding="utf-8")
    print("\n" + table)
    return 0 if len(rows) == len(model_names) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--measure-one", help=argparse.SUPPRESS)
    options = parser.parse_args()
    if options.measure_one:
        print(json.dumps(measure(options.measure_one)), flush=True)
        sys.exit(0)
    sys.exit(main(options.models))
