"""Kaggle GPU job: teacher (bge-m3) scores the v3 question -> passage pairs and mines hard negatives.

For every pair from qgen/build_pairs.py:
1. Agreement: a pair whose chunk was split into several passages is kept only if the teacher also
   ranks the chosen passage above the chunk's other passages (stem matching and teacher agreed 88%
   on a sample, so this drops the uncertain ~12%).
2. Hard negatives: the NEGATIVES highest-scoring passages from OTHER articles (never the same
   article or chunk), skipping any that score within FALSE_NEGATIVE_RATIO of the positive: those
   may answer the question too.
3. Teacher scores for the positive and each negative, which the student's ranking loss imitates.

Uses every GPU it is given. Output (/kaggle/working): scored_pairs.parquet, passages.parquet,
teacher_pairs_info.json. Push from the project root:  uv run kaggle kernels push -p kaggle/teacher_pairs
"""

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import sentence_transformers
import torch
from sentence_transformers import SentenceTransformer

TEACHER = "BAAI/bge-m3"
MAX_TOKENS = 320  # same cap as the student
BATCH_SIZE = 128
CANDIDATES = 64  # teacher top-k per question to pick negatives from
NEGATIVES = 7
FALSE_NEGATIVE_RATIO = 0.95  # a candidate scoring >= 95% of the positive may be a correct answer
SEED = 42

INPUT_ROOT = Path(os.environ.get("TP_INPUT", "/kaggle/input"))
OUT = Path(os.environ.get("TP_OUTPUT", "/kaggle/working"))
LIMIT = int(os.environ.get("TP_LIMIT", "0"))  # >0: smoke test on the first N pairs
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def find_input(name: str) -> Path:
    matches = sorted(INPUT_ROOT.rglob(name))
    if not matches:
        sys.exit(f"❌ {name} not found under {INPUT_ROOT}")
    return matches[0]


def encode_on_all_gpus(model: SentenceTransformer, texts: list[str]) -> torch.Tensor:
    """One worker per GPU (Kaggle gives T4 x2); a single process on one GPU or the CPU."""
    options = {"batch_size": BATCH_SIZE, "normalize_embeddings": True}
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] if DEVICE == "cuda" else []
    if len(devices) <= 1:
        vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=True, **options)
    else:
        pool = model.start_multi_process_pool(target_devices=devices)
        try:
            try:
                vectors = model.encode(texts, pool=pool, convert_to_numpy=True, show_progress_bar=True, **options)
            except TypeError:  # older sentence-transformers: encode() has no pool argument
                vectors = model.encode_multi_process(texts, pool, **options)
        finally:
            model.stop_multi_process_pool(pool)
    return torch.from_numpy(np.asarray(vectors, dtype=np.float16))


def main() -> None:
    started = time.monotonic()
    rng = random.Random(SEED)
    print(f"{torch.cuda.device_count() if DEVICE == 'cuda' else 0} GPU(s) | torch {torch.__version__}"
          f" | sentence-transformers {sentence_transformers.__version__}", flush=True)  # fmt: skip
    pairs = pq.read_table(find_input("train_pairs.parquet")).to_pylist()
    if LIMIT:
        pairs = pairs[:LIMIT]
    pool = list(dict.fromkeys([p["positive"] for p in pairs] + [s for p in pairs for s in p["siblings"]]))
    index = {text: i for i, text in enumerate(pool)}
    groups_of = [set() for _ in pool]  # articles each passage belongs to, to never use one as its own negative
    for p in pairs:
        for text in [p["positive"], *p["siblings"]]:
            groups_of[index[text]].add(p["group"])
    print(f"📄 {len(pairs)} pairs, {len(pool)} distinct passages", flush=True)

    model = SentenceTransformer(TEACHER, device=DEVICE)
    if DEVICE == "cuda":
        model.half()
    model.max_seq_length = MAX_TOKENS
    passage_vectors = encode_on_all_gpus(model, pool)
    query_vectors = encode_on_all_gpus(model, [p["query"] for p in pairs])
    print(f"✅ embedded in {(time.monotonic() - started) / 60:.1f} min", flush=True)

    device = DEVICE if DEVICE == "cpu" else "cuda:0"
    passages_on_device = passage_vectors.to(device)
    compute_dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    passages_on_device = passages_on_device.to(compute_dtype)
    kept, stats = [], {"disagreed_with_siblings": 0, "false_negative_skips": 0, "random_fill": 0}
    for start in range(0, len(pairs), 4096):
        chunk = pairs[start : start + 4096]
        queries = query_vectors[start : start + len(chunk)].to(device).to(compute_dtype)
        scores = queries @ passages_on_device.T  # [chunk, pool]
        top_scores, top_ids = scores.topk(min(CANDIDATES, len(pool)), dim=1)
        scores, top_scores, top_ids = scores.float().cpu(), top_scores.float().cpu(), top_ids.cpu()
        for row, pair in enumerate(chunk):
            positive = index[pair["positive"]]
            positive_score = float(scores[row, positive])
            siblings = [index[s] for s in pair["siblings"]]
            if siblings and float(scores[row, siblings].max()) >= positive_score:
                stats["disagreed_with_siblings"] += 1
                continue
            excluded = {positive, *siblings}
            negatives = []
            for score, candidate in zip(top_scores[row].tolist(), top_ids[row].tolist(), strict=True):
                if candidate in excluded or pair["group"] in groups_of[candidate]:
                    continue
                if score >= FALSE_NEGATIVE_RATIO * positive_score:
                    stats["false_negative_skips"] += 1
                    continue
                negatives.append(candidate)
                if len(negatives) == NEGATIVES:
                    break
            while len(negatives) < NEGATIVES:  # rare: fill with random passages from other articles
                candidate = rng.randrange(len(pool))
                if candidate not in excluded and pair["group"] not in groups_of[candidate] and candidate not in negatives:
                    negatives.append(candidate)
                    stats["random_fill"] += 1
            kept.append({
                "query": pair["query"], "source": pair["source"], "positive_id": positive,
                "negative_ids": negatives, "positive_score": round(positive_score, 4),
                "negative_scores": [round(float(scores[row, n]), 4) for n in negatives],
            })  # fmt: skip
        print(f"   scored {min(start + 4096, len(pairs))}/{len(pairs)}", flush=True)

    pq.write_table(pa.Table.from_pylist(kept), OUT / "scored_pairs.parquet")
    pq.write_table(pa.table({"passage_id": list(range(len(pool))), "text": pool}), OUT / "passages.parquet")
    margins = [k["positive_score"] - max(k["negative_scores"]) for k in kept]
    info = {
        "pairs_in": len(pairs), "pairs_kept": len(kept), "passages": len(pool), **stats,
        "mean_positive_score": round(float(np.mean([k["positive_score"] for k in kept])), 4),
        "mean_hardest_negative_score": round(float(np.mean([max(k["negative_scores"]) for k in kept])), 4),
        "median_margin": round(float(np.median(margins)), 4),
        "gpus": torch.cuda.device_count() if DEVICE == "cuda" else 0,
        "minutes_total": round((time.monotonic() - started) / 60, 1),
    }  # fmt: skip
    (OUT / "teacher_pairs_info.json").write_text(json.dumps(info, indent=2))
    print(f"🏁 {json.dumps(info)}", flush=True)


if __name__ == "__main__":
    main()
