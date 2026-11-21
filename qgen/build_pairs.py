"""Build v3 question -> passage training pairs: uv run python qgen/build_pairs.py

Sources (credit them in the model card):
- tbilisi-ai-lab/wiki-ka-QA (ODC-By): LLM-written questions over Georgian Wikipedia chunks. A native
  speaker rated 41/50 sampled questions good and every sampled question was answered by its chunk.
- PaDaS-Lab/webfaq kat (CC-BY-4.0): real FAQ pairs, minus gambling/SEO spam, at most 25 per website.

Each wiki-ka-QA chunk (~1,500 chars) is longer than the student reads (320 tokens, ~1,000 chars of
Georgian), so it is cut into passages with the same splitter as the distillation texts, and each
question is paired with the passage that holds most of its answer's words. Pairs whose answer is not
mostly inside one passage are dropped, as are pairs overlapping any test set (8-word shingles) and
repeated questions. Writes data/pairs/train_pairs.parquet and data/pairs/build_report.json.
"""

import ast
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from datasets import Dataset
from huggingface_hub import hf_hub_download

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import build_dataset as bd  # noqa: E402  (same splitter, cleaning and test-overlap check)

OUT = PROJECT / "data" / "pairs"
SEED = 42
MIN_ANSWER_COVERAGE = 0.6  # share of the answer's word stems that must appear in the chosen passage
MIN_ANSWER_WORD_CHARS = 4  # ignore short function words when matching answers to passages
STEM_CHARS = 5  # Georgian inflects word endings (ჩხავერი / ჩხავერის): compare the first letters only
WEBFAQ_PER_SITE = 25
WEBFAQ_SPAM = ("casino", "bet", "slot", "poker", "seo.computer", "bonus")
WORD = re.compile(r"\w+")


def stems(text: str) -> set[str]:
    """Rough stems: exact word forms missed matches like ჩხავერი / ჩხავერის, dropping correct pairs."""
    return {w[:STEM_CHARS] for w in WORD.findall(text.casefold()) if len(w) >= MIN_ANSWER_WORD_CHARS}


def best_passage(passages: list[str], answer: str) -> tuple[str | None, float]:
    """The passage holding the largest share of the answer's word stems, and that share. A chunk
    that fits in one passage needs no matching: the question was written from it."""
    if len(passages) == 1:
        return passages[0], 1.0
    wanted = stems(answer)
    if not wanted:
        return None, 0.0
    scored = [(len(wanted & stems(p)) / len(wanted), p) for p in passages]
    coverage, passage = max(scored, key=lambda item: item[0])
    return passage, coverage


def wiki_qa_pairs(stats: Counter) -> list[dict]:
    path = hf_hub_download("tbilisi-ai-lab/wiki-ka-QA", "data/train-00000-of-00001.parquet", repo_type="dataset")
    pairs = []
    for row in pq.read_table(path).to_pylist():
        try:
            qa = ast.literal_eval(row["qa"]) if isinstance(row["qa"], str) else row["qa"]
        except (ValueError, SyntaxError):
            stats["wikiqa: unparsable chunk"] += 1
            continue
        passages = bd.split_passages(row["text"])
        if not passages:
            stats["wikiqa: chunk too short"] += 1
            continue
        for item in qa if isinstance(qa, list) else []:
            question = bd.normalize_spaces(str(item.get("შეკითხვა", ""))) if isinstance(item, dict) else ""
            answer = str(item.get("პასუხი", "")) if isinstance(item, dict) else ""
            if not question:
                stats["wikiqa: empty question"] += 1
                continue
            stats["wikiqa: questions"] += 1
            passage, coverage = best_passage(passages, answer)
            if passage is None or coverage < MIN_ANSWER_COVERAGE:
                stats["wikiqa: answer not inside one passage"] += 1
                continue
            # siblings: the chunk's other passages. The teacher job keeps a multi-passage pair only if
            # bge-m3 also picks `positive` among them (stems and teacher agreed on 88%), and never
            # uses a sibling as a negative.
            pairs.append({"query": question, "positive": passage, "source": "wiki-ka-qa",
                          "group": f"wikiqa:{row['title']}",
                          "siblings": [p for p in passages if p != passage]})  # fmt: skip
    return pairs


def webfaq_pairs(stats: Counter) -> list[dict]:
    path = hf_hub_download("PaDaS-Lab/webfaq", "data/kat.jsonl", repo_type="dataset")
    per_site, pairs = Counter(), []
    for line in open(path, encoding="utf-8"):
        row = json.loads(line)
        question, answer = bd.normalize_spaces(row["question"]), bd.normalize_spaces(row["answer"])
        stats["webfaq: pairs"] += 1
        if any(word in row["origin"].lower() for word in WEBFAQ_SPAM):
            stats["webfaq: spam site"] += 1
        elif len(answer) < 40 or not (bd.good_quality(question, "ka") and bd.good_quality(answer, "ka")):
            stats["webfaq: not clean Georgian or too short"] += 1
        elif per_site[row["origin"]] >= WEBFAQ_PER_SITE:
            stats["webfaq: over per-site cap"] += 1
        else:
            per_site[row["origin"]] += 1
            pairs.append({"query": question, "positive": answer[: bd.MAX_PASSAGE_CHARS], "source": "webfaq",
                          "group": f"webfaq:{row['origin']}", "siblings": []})  # fmt: skip
    return pairs


def main() -> None:
    stats: Counter = Counter()
    tests = bd.test_texts()
    test_keys = {bd.dedup_key(text) for text in tests}
    test_shingles = set().union(*(bd.shingles(text) for text in tests))

    def overlaps_test(text: str) -> bool:
        return bd.dedup_key(text) in test_keys or not bd.shingles(text).isdisjoint(test_shingles)

    kept, seen_questions = [], set()
    for pair in wiki_qa_pairs(stats) + webfaq_pairs(stats):
        key = bd.dedup_key(pair["query"])
        if key in seen_questions:
            stats[f"{pair['source']}: repeated question"] += 1
        elif overlaps_test(pair["query"]) or overlaps_test(pair["positive"]):
            stats[f"{pair['source']}: test overlap"] += 1
        else:
            seen_questions.add(key)
            kept.append(pair)
    random.Random(SEED).shuffle(kept)

    OUT.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(kept).to_parquet(OUT / "train_pairs.parquet")
    report = {
        "pairs": len(kept),
        "by_source": dict(Counter(pair["source"] for pair in kept)),
        "unique_passages": len({pair["positive"] for pair in kept}),
        "filters": dict(sorted(stats.items())),
        "settings": {"min_answer_coverage": MIN_ANSWER_COVERAGE, "webfaq_per_site": WEBFAQ_PER_SITE, "seed": SEED},
    }
    (OUT / "build_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
