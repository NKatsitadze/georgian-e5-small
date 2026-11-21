"""Build the distillation training texts: uv run python build_dataset.py

Sources (all allow training; name them in the model card):
- Georgian Wikipedia (CC BY-SA 3.0 / GFDL): formal text
- FineWeb-2 Georgian (ODC-By): everyday web text
- WebFAQ Georgian (CC-BY-4.0): real questions and their answers
- English Wikipedia (CC BY-SA 3.0 / GFDL): a small share so the student keeps its English

Steps: split documents into passages -> quality filter -> exact dedup -> drop anything that
overlaps a test set (8-word shingles) -> write data/train_texts.parquet + data/build_report.json.
Re-running with the same SEED gives the same file.
"""

import hashlib
import json
import random
import re
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import mteb
import pyarrow.parquet as pq
from datasets import Dataset
from huggingface_hub import hf_hub_download

DATA_DIR = Path(__file__).resolve().parent / "data"
SEED = 42

# Targets: passages kept per source (WebFAQ is used whole). Small enough for a first run.
WIKI_KA_PASSAGES = 150_000
FINEWEB_KA_PASSAGES = 250_000
WIKI_EN_PASSAGES = 30_000
PASSAGES_PER_DOC = 3  # spread the sample over many documents instead of a few long ones

# Fixed source files (~2.5 GB in total, cached by huggingface_hub). FineWeb-2's shards are random
# splits of the same data, so its smaller shard is enough; one English shard is plenty for 30k.
WIKIPEDIA_FILES = {
    "ka": ("20231101.ka/train-00000-of-00002.parquet", "20231101.ka/train-00001-of-00002.parquet"),
    "en": ("20231101.en/train-00001-of-00041.parquet",),
}
FINEWEB_KA_FILES = ("data/kat_Geor/train/001_00000.parquet",)

MIN_PASSAGE_CHARS, MAX_PASSAGE_CHARS = 100, 1000  # ~30-290 tokens of Georgian
MIN_QUESTION_CHARS = 10
MIN_GEORGIAN_RATIO = 0.8  # share of letters in Georgian script
MIN_ENGLISH_RATIO = 0.9
MIN_LETTER_RATIO = 0.6  # drops tables, number lists and markup
SHINGLE_WORDS = 8

TEST_TASKS = ("BelebeleRetrieval", "GeorgianFAQRetrieval", "WebFAQRetrieval")
GEORGIAN_LETTER = re.compile(r"[Ⴀ-ჿᲐ-Ჿⴀ-⴯]")
LATIN_LETTER = re.compile(r"[A-Za-z]")
SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")
WORD = re.compile(r"\w+")


@dataclass(frozen=True)
class Text:
    text: str
    source: str
    lang: str  # "ka" or "en"
    kind: str  # "passage" or "question"


# --- cleaning ------------------------------------------------------------------------------


def normalize_spaces(text: str) -> str:
    return " ".join(text.split())


def split_long(paragraph: str) -> list[str]:
    """Split a paragraph longer than MAX_PASSAGE_CHARS at sentence ends, hard-cutting if needed."""
    pieces = []
    for sentence in SENTENCE_END.split(paragraph):
        while len(sentence) > MAX_PASSAGE_CHARS:
            cut = sentence.rfind(" ", 0, MAX_PASSAGE_CHARS)
            cut = cut if cut > 0 else MAX_PASSAGE_CHARS
            pieces.append(sentence[:cut])
            sentence = sentence[cut:].strip()
        if sentence:
            pieces.append(sentence)
    return pieces


def split_passages(document: str) -> list[str]:
    """Merge consecutive paragraphs into passages of at most MAX_PASSAGE_CHARS."""
    passages: list[str] = []
    current = ""
    for raw in document.split("\n"):
        paragraph = normalize_spaces(raw)
        if not paragraph:
            continue
        pieces = split_long(paragraph) if len(paragraph) > MAX_PASSAGE_CHARS else [paragraph]
        for piece in pieces:
            if current and len(current) + 1 + len(piece) > MAX_PASSAGE_CHARS:
                passages.append(current)
                current = piece
            else:
                current = f"{current} {piece}".strip()
    if current:
        passages.append(current)
    return [p for p in passages if len(p) >= MIN_PASSAGE_CHARS]


def script_ratio(text: str, letters_of_script: re.Pattern[str]) -> float:
    letters = sum(char.isalpha() for char in text)
    return len(letters_of_script.findall(text)) / letters if letters else 0.0


def good_quality(text: str, lang: str) -> bool:
    visible = sum(not char.isspace() for char in text)
    if not visible or sum(char.isalpha() for char in text) / visible < MIN_LETTER_RATIO:
        return False
    if lang == "ka":
        return script_ratio(text, GEORGIAN_LETTER) >= MIN_GEORGIAN_RATIO
    return script_ratio(text, LATIN_LETTER) >= MIN_ENGLISH_RATIO


def dedup_key(text: str) -> str:
    return hashlib.sha1(" ".join(WORD.findall(text.casefold())).encode()).hexdigest()


def shingles(text: str) -> set[int]:
    words = WORD.findall(text.casefold())
    return {hash(tuple(words[i : i + SHINGLE_WORDS])) for i in range(len(words) - SHINGLE_WORDS + 1)}


# --- sources -------------------------------------------------------------------------------


def passages_from(documents: Iterable[str], source: str, lang: str, target: int) -> Iterator[Text]:
    """Up to PASSAGES_PER_DOC good passages per document, until `target` passages."""
    kept = 0
    for document in documents:
        for passage in split_passages(document)[:PASSAGES_PER_DOC]:
            if good_quality(passage, lang):
                yield Text(passage, source, lang, "passage")
                kept += 1
                if kept >= target:
                    return


def shuffled_texts(repo: str, files: tuple[str, ...]) -> Iterator[str]:
    """Documents from downloaded parquet files in a seeded random order: row groups are visited
    randomly and shuffled inside, so a sample spreads over the whole file without loading it all.
    Downloads are cached by huggingface_hub; streaming was dropped because its background
    threads kept the process alive after the last row was read."""
    rng = random.Random(SEED)
    groups = []
    for name in files:
        parquet = pq.ParquetFile(hf_hub_download(repo, name, repo_type="dataset"))
        groups += [(parquet, index) for index in range(parquet.num_row_groups)]
    rng.shuffle(groups)
    for parquet, index in groups:
        texts = parquet.read_row_group(index, columns=["text"]).column("text").to_pylist()
        rng.shuffle(texts)
        yield from texts


def wikipedia(lang: str, target: int) -> Iterator[Text]:
    return passages_from(shuffled_texts("wikimedia/wikipedia", WIKIPEDIA_FILES[lang]), f"wikipedia-{lang}", lang, target)


def fineweb_georgian(target: int) -> Iterator[Text]:
    return passages_from(shuffled_texts("HuggingFaceFW/fineweb-2", FINEWEB_KA_FILES), "fineweb2-ka", "ka", target)


def webfaq_georgian() -> Iterator[Text]:
    path = hf_hub_download("PaDaS-Lab/webfaq", "data/kat.jsonl", repo_type="dataset")
    with open(path, encoding="utf-8") as lines:
        for line in lines:
            row = json.loads(line)
            question = normalize_spaces(row["question"])
            if len(question) >= MIN_QUESTION_CHARS and good_quality(question, "ka"):
                yield Text(question, "webfaq-ka", "ka", "question")
            answer = normalize_spaces(row["answer"])[:MAX_PASSAGE_CHARS]
            if len(answer) >= MIN_QUESTION_CHARS and good_quality(answer, "ka"):
                yield Text(answer, "webfaq-ka", "ka", "passage")


# --- test-set overlap ----------------------------------------------------------------------


def _document_text(row: dict) -> str:
    return f"{row.get('title') or ''} {row['text']}".strip()


def test_texts() -> list[str]:
    """Every query and document of the Georgian test subsets (Belebele includes English sides).

    mteb 2.x tasks use one of two layouts: task.dataset[subset][split] with row lists, or (older
    tasks such as Belebele) task.queries / task.corpus as {subset: {split: {id: text-or-row}}}.
    """
    texts: list[str] = []
    for name in TEST_TASKS:
        task = mteb.get_task(name, languages=["kat"])
        task.load_data()
        if task.dataset is not None:
            for subset in task.dataset.values():
                for split in subset.values():
                    texts += [row["text"] for row in split["queries"]]
                    texts += [_document_text(row) for row in split["corpus"]]
        else:
            for splits in task.queries.values():
                for queries in splits.values():
                    texts += [q if isinstance(q, str) else q["text"] for q in queries.values()]
            for splits in task.corpus.values():
                for corpus in splits.values():
                    texts += [_document_text(row) for row in corpus.values()]
        print(f"   test texts after {name}: {len(texts)}", flush=True)
    if not texts:
        raise RuntimeError("no test texts loaded: the overlap filter would silently do nothing")
    return texts


@dataclass
class SourceReport:
    collected: int = 0
    duplicates: int = 0
    test_overlap: int = 0
    kept: int = 0


def build() -> dict:
    started = time.monotonic()
    DATA_DIR.mkdir(exist_ok=True)

    tests = test_texts()
    test_keys = {dedup_key(text) for text in tests}
    test_shingles = set().union(*(shingles(text) for text in tests))
    print(f"🧪 test sets: {len(tests)} texts, {len(test_shingles)} shingles to exclude", flush=True)

    sources = (
        ("webfaq-ka", webfaq_georgian),
        ("wikipedia-ka", lambda: wikipedia("ka", WIKI_KA_PASSAGES)),
        ("fineweb2-ka", lambda: fineweb_georgian(FINEWEB_KA_PASSAGES)),
        ("wikipedia-en", lambda: wikipedia("en", WIKI_EN_PASSAGES)),
    )
    reports: dict[str, SourceReport] = {}
    seen: set[str] = set()
    kept: list[Text] = []
    for name, load in sources:
        report = reports[name] = SourceReport()
        print(f"▶ {name}", flush=True)
        for item in load():
            report.collected += 1
            key = dedup_key(item.text)
            if key in seen:
                report.duplicates += 1
                continue
            seen.add(key)
            if key in test_keys or not shingles(item.text).isdisjoint(test_shingles):
                report.test_overlap += 1
                continue
            kept.append(item)
            report.kept += 1
            if report.collected % 50_000 == 0:
                print(f"   {name}: {report.collected} collected, {report.kept} kept", flush=True)
        print(f"✅ {name}: {asdict(report)}", flush=True)

    random.Random(SEED).shuffle(kept)
    Dataset.from_list([asdict(item) for item in kept]).to_parquet(DATA_DIR / "train_texts.parquet")

    lengths = sorted(len(item.text) for item in kept)
    summary = {
        "total_texts": len(kept),
        "by_source": {name: asdict(report) for name, report in reports.items()},
        "by_kind": dict(Counter(item.kind for item in kept)),
        "by_lang": dict(Counter(item.lang for item in kept)),
        "median_chars": lengths[len(lengths) // 2] if lengths else 0,
        "test_texts_excluded_against": len(tests),
        "source_files": {"wikipedia": WIKIPEDIA_FILES, "fineweb2": FINEWEB_KA_FILES, "webfaq": "data/kat.jsonl"},
        "seed": SEED,
        "minutes": round((time.monotonic() - started) / 60, 1),
    }
    (DATA_DIR / "build_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
