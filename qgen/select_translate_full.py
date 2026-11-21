"""Pick the Georgian questions to translate into English for v5 (English question -> Georgian passage).

    uv run python qgen/select_translate_full.py
    uv run kaggle datasets create -p data/kaggle/translate-full     # first time

One question per passage (the most varied set for the budget), from wiki-ka-QA pairs only. The
translation pilot's 100 questions are left out, so they stay a clean review sample.
Output: to_translate.parquet with id, query (Georgian), passage, group.
"""

import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "data" / "kaggle" / "translate-full"
SEED = 13
QUESTIONS = 30_000


def main() -> None:
    pilot = {json.loads(line)["query"] for line in open(PROJECT / "data" / "kaggle" / "translate-pilot" / "pilot_pairs.jsonl", encoding="utf-8")}
    rows = [r for r in pq.read_table(PROJECT / "data" / "pairs" / "train_pairs.parquet").to_pylist()
            if r["source"] == "wiki-ka-qa" and r["query"] not in pilot]  # fmt: skip
    random.Random(SEED).shuffle(rows)
    picked, seen_passages = [], set()
    for row in rows:
        if row["positive"] not in seen_passages:
            picked.append(row)
            seen_passages.add(row["positive"])
        if len(picked) == QUESTIONS:
            break
    OUT.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "id": [f"q{i:05d}" for i in range(len(picked))],
        "query": [r["query"] for r in picked],
        "passage": [r["positive"] for r in picked],
        "group": [r["group"] for r in picked],
    })  # fmt: skip
    pq.write_table(table, OUT / "to_translate.parquet")
    metadata = {
        "title": "Georgian questions to translate",
        "id": "nikakatsitadze/georgian-translate-full",
        "subtitle": "30k wiki-ka-QA questions with passages, for English translation",
        "description": "Questions and passages from tbilisi-ai-lab/wiki-ka-QA (ODC-By), one question per passage, "
        "to be translated into English with Qwen3-14B (Apache-2.0).",
        "licenses": [{"name": "other"}],
    }
    (OUT / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"✅ {len(picked)} questions ({len(rows)} candidates) -> {OUT}")


if __name__ == "__main__":
    main()
