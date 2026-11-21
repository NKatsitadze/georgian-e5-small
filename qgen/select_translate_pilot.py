"""Pick Georgian questions for the translation pilot (v5 idea: English questions -> Georgian passages).

    uv run python qgen/select_translate_pilot.py
    uv run kaggle datasets create -p data/kaggle/translate-pilot

- pilot_pairs.jsonl: 100 wiki-ka-QA question -> passage pairs (one per article) whose questions the
  LLMs translate into English.
- distractors.jsonl: 5,000 passages from other articles. The teacher must find each English
  question's Georgian passage among them (round-trip check), the same check as the Georgian original.
"""

import json
import random
from pathlib import Path

import pyarrow.parquet as pq

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "data" / "kaggle" / "translate-pilot"
SEED = 11
PILOT = 100
DISTRACTORS = 5_000


def main() -> None:
    rows = [r for r in pq.read_table(PROJECT / "data" / "pairs" / "train_pairs.parquet").to_pylist()
            if r["source"] == "wiki-ka-qa"]  # fmt: skip
    rng = random.Random(SEED)
    rng.shuffle(rows)
    pilot, seen_groups = [], set()
    for row in rows:
        if row["group"] not in seen_groups:
            pilot.append(row)
            seen_groups.add(row["group"])
        if len(pilot) == PILOT:
            break
    others = list({r["positive"]: r for r in rows if r["group"] not in seen_groups}.values())
    distractors = rng.sample(others, DISTRACTORS)

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "pilot_pairs.jsonl", "w", encoding="utf-8") as f:
        for index, row in enumerate(pilot):
            f.write(json.dumps({"id": f"t{index:03d}", "query": row["query"], "passage": row["positive"]}, ensure_ascii=False) + "\n")
    with open(OUT / "distractors.jsonl", "w", encoding="utf-8") as f:
        for row in distractors:
            f.write(json.dumps({"text": row["positive"]}, ensure_ascii=False) + "\n")
    metadata = {
        "title": "Georgian question translation pilot",
        "id": "nikakatsitadze/georgian-translate-pilot",
        "subtitle": "100 Georgian questions + 5k distractor passages for a translation pilot",
        "description": "Questions and passages from tbilisi-ai-lab/wiki-ka-QA (ODC-By), for testing which open "
        "LLM translates Georgian search questions into English well.",
        "licenses": [{"name": "other"}],
    }
    (OUT / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"✅ {len(pilot)} pilot pairs, {len(distractors)} distractors -> {OUT}")


if __name__ == "__main__":
    main()
