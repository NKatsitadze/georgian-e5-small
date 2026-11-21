"""Pick passages for the question-generation pilot and package them as a private Kaggle dataset.

    uv run python qgen/select_pilot.py
    uv run kaggle datasets create -p data/kaggle/qgen-pilot     # first time
    uv run kaggle datasets version -p data/kaggle/qgen-pilot -m "..."   # later

- pilot_passages.jsonl: 50 Georgian passages (25 Wikipedia, 25 FineWeb-2) that the LLMs write
  questions for. Only clean-licensed sources from train_texts.parquet (already test-filtered).
- distractors.jsonl: 5,000 other Georgian passages. The teacher must find each question's source
  passage among them (round-trip check), which shows whether a question is specific and answerable.
"""

import json
import random
from pathlib import Path

import pyarrow.parquet as pq

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "data" / "kaggle" / "qgen-pilot"
SEED = 7
PER_SOURCE = 25
DISTRACTORS = 5_000
MIN_CHARS, MAX_CHARS = 300, 1000  # enough content to ask about, short enough to read in a review


SENTENCE_END = (".", "!", "?", "…", "»", "“", '"')
MAX_SYMBOL_SHARE = 0.12  # digits, brackets and list punctuation: lists of titles/years read badly


def well_formed(text: str) -> bool:
    """A passage worth asking about: starts at a sentence start, ends at a sentence end, and is
    prose rather than a list (e.g. '…თეატრი); „…“ (1971), …' started inside a parenthesis)."""
    first_close, first_open = text.find(")"), text.find("(")
    if first_close != -1 and (first_open == -1 or first_close < first_open):
        return False  # began in the middle of a parenthesis
    if not text[0].isalpha() or not text.rstrip().endswith(SENTENCE_END):
        return False
    symbols = sum(char.isdigit() or char in "()[];:,„“\"«»/–-" for char in text)
    return symbols / len(text) <= MAX_SYMBOL_SHARE


def main() -> None:
    rows = pq.read_table(PROJECT / "data" / "train_texts.parquet").to_pylist()
    georgian = [r for r in rows if r["lang"] == "ka" and r["kind"] == "passage"]
    rng = random.Random(SEED)

    pilot = []
    for source in ("wikipedia-ka", "fineweb2-ka"):
        candidates = [
            r for r in georgian
            if r["source"] == source and MIN_CHARS <= len(r["text"]) <= MAX_CHARS and well_formed(r["text"])
        ]  # fmt: skip
        print(f"{source}: {len(candidates)} well-formed candidates")
        pilot += rng.sample(candidates, PER_SOURCE)
    pilot_texts = {r["text"] for r in pilot}
    distractors = rng.sample([r for r in georgian if r["text"] not in pilot_texts], DISTRACTORS)

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "pilot_passages.jsonl", "w", encoding="utf-8") as f:
        for index, row in enumerate(pilot):
            f.write(json.dumps({"id": f"p{index:03d}", "source": row["source"], "text": row["text"]}, ensure_ascii=False) + "\n")
    with open(OUT / "distractors.jsonl", "w", encoding="utf-8") as f:
        for row in distractors:
            f.write(json.dumps({"source": row["source"], "text": row["text"]}, ensure_ascii=False) + "\n")
    metadata = {
        "title": "Georgian question generation pilot",
        "id": "nikakatsitadze/georgian-qgen-pilot",
        "subtitle": "50 passages + 5k distractors for a question-generation pilot",
        "description": "Georgian passages from Georgian Wikipedia (CC BY-SA 3.0 / GFDL) and FineWeb-2 kat_Geor "
        "(ODC-By), for testing which open LLM writes good Georgian search questions.",
        "licenses": [{"name": "other"}],
    }
    (OUT / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"✅ {len(pilot)} pilot passages, {len(distractors)} distractors -> {OUT}")


if __name__ == "__main__":
    main()
