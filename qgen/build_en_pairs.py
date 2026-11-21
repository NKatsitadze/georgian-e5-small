"""Turn Qwen3-14B English translations of wiki-ka-QA questions into v5 training pairs.

    uv run python qgen/build_en_pairs.py data/qgen/translate-full/translations.parquet
    uv run kaggle datasets create -p data/kaggle/train-pairs-v5      # first time

Each English question keeps its Georgian question's passage, article (group) and sibling
passages, so the teacher's agreement check and hard-negative mining work exactly as for v3.
Filter (tested against the round-3 human review): drop translations that are empty or contain
non-Latin script (CJK, Georgian, Cyrillic, Arabic...), e.g. "Which clubs has内马尔 played for".
A similarity threshold was tested too and rejected: it dropped more good translations than bad.

Output: data/kaggle/train-pairs-v5/train_pairs.parquet = the v3 pairs unchanged, then the English
pairs appended (source "wiki-ka-qa-en"), so the teacher reproduces v3's Georgian results exactly.
"""

import json
import re
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "data" / "kaggle" / "train-pairs-v5"
# Latin letters with accents, digits, punctuation, typographic quotes/dashes, and symbols like ² or °.
ALLOWED = re.compile(r"^[\x20-\x7E -ɏḀ-ỿ‐-‧‰-⁞²³¹⁰-₟]*$")


def keep(english: str) -> bool:
    return bool(english) and ALLOWED.match(english) is not None


def main() -> None:
    translations = pq.read_table(sys.argv[1]).to_pylist()
    pairs = pq.read_table(PROJECT / "data" / "pairs" / "train_pairs.parquet").to_pylist()
    to_translate = {r["id"]: r for r in pq.read_table(PROJECT / "data" / "kaggle" / "translate-full" / "to_translate.parquet").to_pylist()}
    by_question = {(p["query"], p["positive"]): p for p in pairs}

    english, dropped = [], []
    for t in translations:
        source = by_question[(t["query"], to_translate[t["id"]]["passage"])]
        if not keep(t["en"]):
            dropped.append(t)
            continue
        english.append({**source, "query": t["en"], "source": "wiki-ka-qa-en"})

    OUT.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(pairs + english, schema=pq.read_schema(PROJECT / "data" / "pairs" / "train_pairs.parquet")),
                   OUT / "train_pairs.parquet")  # fmt: skip
    report = {"georgian_pairs": len(pairs), "translations": len(translations), "english_kept": len(english),
              "english_dropped": len(dropped), "dropped_examples": [d["en"] for d in dropped[:20]]}  # fmt: skip
    (OUT / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata = {
        "title": "Georgian embedding train pairs v5",
        "id": "nikakatsitadze/georgian-embedding-train-pairs-v5",
        "subtitle": "v3 Georgian question-passage pairs plus English translations",
        "description": "wiki-ka-QA (tbilisi-ai-lab, ODC-By) and WebFAQ (CC-BY-4.0) question -> passage pairs, plus English "
        "translations of wiki-ka-QA questions made with Qwen3-14B (Apache-2.0).",
        "licenses": [{"name": "other"}],
    }
    (OUT / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "dropped_examples"}), "\ndropped:", report["dropped_examples"][:5])


if __name__ == "__main__":
    main()
