"""Step 1: measure how well existing embedding models find Georgian text.

    uv run python baseline.py                      # every model in MODELS
    uv run python baseline.py --models intfloat/multilingual-e5-small
    uv run python baseline.py --models Snowflake/snowflake-arctic-embed-l-v2.0 --device cpu
    uv run python baseline.py --models data/models/distill-v2/student-epoch6     # a local folder
    uv run python baseline.py --models Vibels/georgian-e5-finetuned Vibels/georgian-e5-finetuned::e5
                                   # any Hub model; ::e5 adds e5's query:/passage: prefixes

Runs the Georgian subsets of MTEB's retrieval tasks and writes results/baseline.md and
results/baseline.json. Raw MTEB results are cached in results/mteb/, so a re-run only
evaluates models or tasks that are missing.
"""

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mteb
from mteb.cache import ResultCache
from sentence_transformers import SentenceTransformer

RESULTS_DIR = Path(__file__).resolve().parent / "results"
GEORGIAN = "kat"

# Open models with a permissive license (possible starting points for fine-tuning) come first.
MODELS = (
    "intfloat/multilingual-e5-small",
    "intfloat/multilingual-e5-base",
    "intfloat/multilingual-e5-large",
    "BAAI/bge-m3",
    "Alibaba-NLP/gte-multilingual-base",
    "Snowflake/snowflake-arctic-embed-l-v2.0",
    "ibm-granite/granite-embedding-97m-multilingual-r2",
    "Qwen/Qwen3-Embedding-0.6B",
)

# Belebele: reading-comprehension passages; its kat-eng / eng-kat subsets test cross-language search.
# GeorgianFAQ and WebFAQ: real questions and answers from Georgian websites.
TASKS = ("BelebeleRetrieval", "GeorgianFAQRetrieval", "WebFAQRetrieval")
METRICS = ("ndcg_at_10", "recall_at_1", "recall_at_10", "mrr_at_10")


@dataclass(frozen=True)
class Score:
    model: str
    task: str
    subset: str
    ndcg_at_10: float
    recall_at_1: float
    recall_at_10: float
    mrr_at_10: float


def georgian_tasks() -> list:
    """Each task restricted to its Georgian subsets (Belebele has 122 languages)."""
    return [mteb.get_task(name, languages=[GEORGIAN]) for name in TASKS]


def extract_scores(model_name: str, model_result) -> list[Score]:
    scores = []
    for task_result in model_result.task_results:
        for split_scores in task_result.scores.values():
            for item in split_scores:
                subset = str(item.get("hf_subset", "default"))
                if GEORGIAN not in subset and subset != "default":
                    continue
                scores.append(
                    Score(model_name, task_result.task_name, subset, *(float(item[m]) for m in METRICS))
                )
    return scores


E5_PROMPTS = {"query": "query: ", "document": "passage: "}


def in_mteb_registry(name: str) -> bool:
    try:
        mteb.get_model_meta(name)
        return True
    except Exception:
        return False


def load_model(model_name: str, device: str | None) -> tuple[object, str, str]:
    """Returns (model, display name, overwrite strategy).

    Registry names go through mteb.get_model (it knows each model's prefixes). A local folder is
    loaded with sentence-transformers and uses the prompts saved with it. mteb would name every
    local model "no_model_name/available", so two of them would share cached results: give each a
    name from its folder and always re-evaluate, since a retrained model can reuse the same path.
    """
    name, _, variant = model_name.partition("::")
    path = Path(name)
    if not path.exists():
        if not variant and in_mteb_registry(name):
            return mteb.get_model(name, device=device), name, "only-missing"
        # Not in mteb's registry (e.g. community Georgian fine-tunes): load it from the Hub as is.
        # "::e5" adds e5's query:/passage: prefixes for models that don't record whether they need them.
        # "::noprefix" forces no prompts at all (e.g. to score e5 without its official prefixes).
        model = SentenceTransformer(name, device=device)
        labels = {"e5": f"{name} (+e5 prefixes)", "noprefix": f"{name} (no prefixes)"}
        if variant == "e5":
            model.prompts = E5_PROMPTS
        elif variant == "noprefix":
            model.prompts = {}
        model.model_card_data.model_name = name + (f"-{variant}" if variant else "")
        return model, labels.get(variant, name), "only-missing"
    model = SentenceTransformer(str(path), device=device)
    label = f"{path.parent.name}/{path.name}"
    model.model_card_data.model_name = f"local/{label}"
    return model, label, "always"


def evaluate_model(model_name: str, tasks: list, cache: ResultCache, device: str | None) -> list[Score]:
    started = time.monotonic()
    print(f"▶ {model_name} (device: {device or 'auto'})", flush=True)
    model, label, overwrite = load_model(model_name, device)
    result = mteb.evaluate(model, tasks, cache=cache, overwrite_strategy=overwrite, show_progress_bar=False)
    scores = extract_scores(label, result)
    print(f"✅ {label}: {len(scores)} subset score(s) in {time.monotonic() - started:.0f}s", flush=True)
    return scores


def to_markdown(scores: list[Score]) -> str:
    """One row per model, nDCG@10 per test plus the average; models sorted by the average."""
    columns = sorted({(s.task, s.subset) for s in scores})
    by_model: dict[str, dict[tuple[str, str], float]] = {}
    for s in scores:
        by_model.setdefault(s.model, {})[(s.task, s.subset)] = s.ndcg_at_10

    def average(row: dict) -> float:
        return sum(row.values()) / len(row) if row else 0.0

    header = ["model", "avg", *(f"{task.removesuffix('Retrieval')} {subset}" for task, subset in columns)]
    lines = [
        "nDCG@10 (0-100, higher is better) on the Georgian subsets of MTEB retrieval tasks.",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for model, row in sorted(by_model.items(), key=lambda item: average(item[1]), reverse=True):
        cells = [f"{row[c] * 100:.1f}" if c in row else "—" for c in columns]
        lines.append(f"| {model} | {average(row) * 100:.1f} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def previous_scores(skip_models: list[str]) -> list[Score]:
    """Scores of models not re-run now, so running one model keeps the rest of the table."""
    path = RESULTS_DIR / "baseline.json"
    if not path.exists():
        return []
    return [Score(**row) for row in json.loads(path.read_text(encoding="utf-8")) if row["model"] not in skip_models]


def main(model_names: list[str], device: str | None) -> int:
    RESULTS_DIR.mkdir(exist_ok=True)
    cache = ResultCache(cache_path=RESULTS_DIR / "mteb")
    tasks = georgian_tasks()
    local_labels = [f"{Path(n).parent.name}/{Path(n).name}" for n in model_names if Path(n).exists()]
    scores: list[Score] = previous_scores(skip_models=model_names + local_labels)
    failed: list[str] = []
    for name in model_names:
        try:
            scores += evaluate_model(name, tasks, cache, device)
        except Exception as error:  # one broken model must not lose the others' results
            print(f"❌ {name}: {type(error).__name__}: {error}", flush=True)
            failed.append(name)

    (RESULTS_DIR / "baseline.json").write_text(
        json.dumps([asdict(s) for s in scores], indent=2), encoding="utf-8"
    )
    table = to_markdown(scores)
    (RESULTS_DIR / "baseline.md").write_text(table, encoding="utf-8")
    print("\n" + table)
    print(f"Done: {len(model_names) - len(failed)} model(s) ok, {len(failed)} failed {failed or ''}")
    return 1 if failed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    # Default: auto (the Mac GPU via MPS). Use cpu for a model that hits an MPS bug.
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    options = parser.parse_args()
    sys.exit(main(options.models, options.device))
