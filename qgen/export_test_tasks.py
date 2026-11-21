"""Export the Georgian MTEB retrieval test sets (queries, corpus, relevance labels) to one JSON file,
so a Kaggle job can evaluate rerankers without depending on the mteb library.

    uv run python qgen/export_test_tasks.py
    uv run kaggle datasets create -p data/kaggle/rerank-test      # first time

This is evaluation data only; it is never used for training.
"""

import json
from pathlib import Path

import mteb

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "data" / "kaggle" / "rerank-test"
TASKS = ("BelebeleRetrieval", "GeorgianFAQRetrieval", "WebFAQRetrieval")


def text_of(row) -> str:
    if isinstance(row, str):
        return row
    title, text = row.get("title") or "", row.get("text") or ""
    return f"{title} {text}".strip() if title else text


def main() -> None:
    subsets = []
    for name in TASKS:
        task = mteb.get_task(name, languages=["kat"])
        task.load_data()
        if task.dataset is not None:
            for subset, splits in task.dataset.items():
                d = splits["test"]
                subsets.append({"task": name, "subset": subset,
                                "queries": {r["id"]: r["text"] for r in d["queries"]},
                                "corpus": {r["id"]: text_of(r) for r in d["corpus"]},
                                "qrels": d["relevant_docs"]})  # fmt: skip
        else:
            for subset in task.queries:
                subsets.append({"task": name, "subset": subset,
                                "queries": {k: text_of(v) for k, v in task.queries[subset]["test"].items()},
                                "corpus": {k: text_of(v) for k, v in task.corpus[subset]["test"].items()},
                                "qrels": task.relevant_docs[subset]["test"]})  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "test_tasks.json").write_text(json.dumps(subsets, ensure_ascii=False), encoding="utf-8")
    metadata = {
        "title": "Georgian retrieval test sets",
        "id": "nikakatsitadze/georgian-rerank-test",
        "subtitle": "Georgian MTEB retrieval test sets exported for reranker evaluation",
        "description": "Georgian subsets of MTEB BelebeleRetrieval, GeorgianFAQRetrieval and WebFAQRetrieval (evaluation only).",
        "licenses": [{"name": "other"}],
    }
    (OUT / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for s in subsets:
        print(f"{s['task']} {s['subset']}: {len(s['queries'])} queries, {len(s['corpus'])} docs")


if __name__ == "__main__":
    main()
