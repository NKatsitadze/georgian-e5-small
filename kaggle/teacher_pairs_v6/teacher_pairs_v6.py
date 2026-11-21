"""Kaggle GPU job (v6): the reranker bge-reranker-v2-m3 becomes the ranking teacher.

Reranker test on the Georgian benchmark: re-sorting bge-m3's top 20 raised nDCG@10 from 85.4 to 89.8,
so its judgements of "which passage answers this question" are better than bge-m3's.

For every v3 question -> passage pair (same input file as v3):
1. bge-m3 proposes candidates: its top CANDIDATES passages from OTHER articles (never the question's
   own article or chunk); the first RERANK_CANDIDATES of them go to the reranker.
2. The reranker scores the positive, the chunk's sibling passages and those candidates.
3. Agreement: the pair is kept only if the reranker ranks the positive above every sibling.
4. Hard negatives: the NEGATIVES highest-scoring candidates, skipping any the reranker scores within
   FALSE_NEGATIVE_MARGIN logits of the positive (they may answer the question too).
5. Both teachers' scores are saved: reranker logits (the new targets) and bge-m3 cosines (so the
   training job can match the target sharpness to v3's and change only WHICH passages are preferred).

Both GPUs: bge-m3 embeds on both, then one reranker process per GPU scores half of the pairs.
Output (/kaggle/working): scored_pairs.parquet, passages.parquet, teacher_pairs_info.json.
Push from the project root:  uv run kaggle kernels push -p kaggle/teacher_pairs_v6
"""

import json
import os
import pickle
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

RETRIEVER = "BAAI/bge-m3"
RERANKER = os.environ.get("TP_RERANKER", "BAAI/bge-reranker-v2-m3")  # Apache-2.0
MAX_TOKENS = 320  # bge-m3 cap, same as the student
RERANK_MAX_LENGTH = 512
CANDIDATES = 64
RERANK_CANDIDATES = 15
NEGATIVES = 7
FALSE_NEGATIVE_MARGIN = 1.0  # skip candidates scoring >= positive logit - 1.0 (odds within ~2.7x)
RERANK_BATCH = 64
SEED = 42

INPUT_ROOT = Path(os.environ.get("TP_INPUT", "/kaggle/input"))
OUT = Path(os.environ.get("TP_OUTPUT", "/kaggle/working"))
WORK = Path(os.environ.get("TP_WORK", "/tmp/tp_v6"))  # intermediate files, kept out of the outputs
LIMIT = int(os.environ.get("TP_LIMIT", "0"))


def find_input(name: str) -> Path:
    matches = sorted(INPUT_ROOT.rglob(name))
    if not matches:
        sys.exit(f"❌ {name} not found under {INPUT_ROOT}")
    return matches[0]


def encode_on_all_gpus(model, texts: list[str]) -> torch.Tensor:
    options = {"batch_size": 128, "normalize_embeddings": True}
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    if len(devices) <= 1:
        vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False, **options)
    else:
        pool = model.start_multi_process_pool(target_devices=devices)
        try:
            try:
                vectors = model.encode(texts, pool=pool, convert_to_numpy=True, show_progress_bar=False, **options)
            except TypeError:
                vectors = model.encode_multi_process(texts, pool, **options)
        finally:
            model.stop_multi_process_pool(pool)
    return torch.from_numpy(np.asarray(vectors, dtype=np.float16))


def propose() -> None:
    """Stage 1 (main process): bge-m3 candidates for every pair -> WORK/proposals.pkl."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from sentence_transformers import SentenceTransformer

    started = time.monotonic()
    pairs = pq.read_table(find_input("train_pairs.parquet")).to_pylist()
    if LIMIT:
        pairs = pairs[:LIMIT]
    pool = list(dict.fromkeys([p["positive"] for p in pairs] + [s for p in pairs for s in p["siblings"]]))
    index = {text: i for i, text in enumerate(pool)}
    groups_of = [set() for _ in pool]
    for p in pairs:
        for text in [p["positive"], *p["siblings"]]:
            groups_of[index[text]].add(p["group"])
    print(f"📄 {len(pairs)} pairs, {len(pool)} distinct passages", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(RETRIEVER, device=device)
    if device == "cuda":
        model.half()
    model.max_seq_length = MAX_TOKENS
    passage_vectors = encode_on_all_gpus(model, pool)
    query_vectors = encode_on_all_gpus(model, [p["query"] for p in pairs])
    print(f"✅ bge-m3 embedded in {(time.monotonic() - started) / 60:.1f} min", flush=True)

    target = "cuda:0" if device == "cuda" else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    passages_on_device = passage_vectors.to(target).to(dtype)
    proposals = []
    for start in range(0, len(pairs), 4096):
        chunk = pairs[start : start + 4096]
        q = query_vectors[start : start + len(chunk)].to(target).to(dtype)
        scores = (q @ passages_on_device.T).float()
        top_ids = scores.topk(min(CANDIDATES, len(pool)), dim=1).indices.cpu().tolist()
        scores = scores.cpu()
        for row, pair in enumerate(chunk):
            positive = index[pair["positive"]]
            siblings = [index[s] for s in pair["siblings"]]
            excluded = {positive, *siblings}
            cands = [c for c in top_ids[row] if c not in excluded and pair["group"] not in groups_of[c]][:RERANK_CANDIDATES]
            proposals.append({"query": pair["query"], "source": pair["source"], "positive": positive, "siblings": siblings,
                              "candidates": cands, "cos": {c: round(float(scores[row, c]), 4) for c in [positive, *cands]}})  # fmt: skip
    WORK.mkdir(parents=True, exist_ok=True)
    with open(WORK / "proposals.pkl", "wb") as f:
        pickle.dump({"proposals": proposals, "pool": pool}, f)
    pq.write_table(pa.table({"passage_id": list(range(len(pool))), "text": pool}), OUT / "passages.parquet")
    print(f"✅ proposals ready in {(time.monotonic() - started) / 60:.1f} min", flush=True)


def rerank_shard(shard: int, shards: int) -> None:
    """Stage 2 (one process per GPU): reranker logits for every (question, passage) of this shard."""
    from sentence_transformers import CrossEncoder

    started = time.monotonic()
    with open(WORK / "proposals.pkl", "rb") as f:
        data = pickle.load(f)
    proposals, pool = data["proposals"][shard::shards], data["pool"]
    jobs = [(n, pid) for n, p in enumerate(proposals) for pid in dict.fromkeys([p["positive"], *p["siblings"], *p["candidates"]])]
    jobs.sort(key=lambda j: len(proposals[j[0]]["query"]) + len(pool[j[1]]))  # similar lengths per batch: less padding
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CrossEncoder(RERANKER, max_length=RERANK_MAX_LENGTH, device=device,
                         model_kwargs={"torch_dtype": torch.float16} if device == "cuda" else {})  # fmt: skip
    def raw_logits(texts: list[tuple[str, str]]) -> np.ndarray:
        try:
            return np.asarray(model.predict(texts, batch_size=RERANK_BATCH, activation_fn=torch.nn.Identity(), show_progress_bar=False))
        except TypeError:  # older sentence-transformers: no activation_fn argument -> sigmoid scores, invert them
            p = np.clip(np.asarray(model.predict(texts, batch_size=RERANK_BATCH, show_progress_bar=False), dtype=np.float64), 1e-7, 1 - 1e-7)
            return np.log(p / (1 - p))

    logits = np.empty(len(jobs), dtype=np.float32)
    step = 50_000
    for start in range(0, len(jobs), step):
        batch = jobs[start : start + step]
        logits[start : start + len(batch)] = raw_logits([(proposals[n]["query"], pool[pid]) for n, pid in batch])
        rate = (start + len(batch)) / (time.monotonic() - started)
        print(f"   GPU {shard}: {start + len(batch)}/{len(jobs)} pairs | {rate:.0f}/s | ~{(len(jobs) - start - len(batch)) / rate / 60:.0f} min left", flush=True)
    scores: list[dict] = [{} for _ in proposals]
    for (n, pid), value in zip(jobs, logits.tolist(), strict=True):
        scores[n][pid] = value
    with open(WORK / f"scores_{shard}.pkl", "wb") as f:
        pickle.dump(scores, f)


def select(shards: int) -> None:
    """Stage 3 (main process): agreement check, negatives, output."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    with open(WORK / "proposals.pkl", "rb") as f:
        proposals = pickle.load(f)["proposals"]
    scores = [None] * len(proposals)
    for shard in range(shards):
        with open(WORK / f"scores_{shard}.pkl", "rb") as f:
            for offset, value in enumerate(pickle.load(f)):
                scores[shard + offset * shards] = value
    rng = random.Random(SEED)
    kept, stats = [], {"disagreed_with_siblings": 0, "false_negative_skips": 0, "short_of_negatives": 0}
    for p, s in zip(proposals, scores, strict=True):
        positive = s[p["positive"]]
        if p["siblings"] and max(s[x] for x in p["siblings"]) >= positive:
            stats["disagreed_with_siblings"] += 1
            continue
        ranked = sorted(p["candidates"], key=lambda c: -s[c])
        negatives = []
        for c in ranked:
            if s[c] >= positive - FALSE_NEGATIVE_MARGIN:
                stats["false_negative_skips"] += 1
                continue
            negatives.append(c)
            if len(negatives) == NEGATIVES:
                break
        if len(negatives) < NEGATIVES:  # rare: fewer safe candidates than needed
            stats["short_of_negatives"] += 1
            continue
        rng.shuffle(negatives)  # order carries no information
        kept.append({"query": p["query"], "source": p["source"], "positive_id": p["positive"], "negative_ids": negatives,
                     "positive_score": round(positive, 4), "negative_scores": [round(s[c], 4) for c in negatives],
                     "positive_cos": p["cos"][p["positive"]], "negative_cos": [p["cos"][c] for c in negatives]})  # fmt: skip
    pq.write_table(pa.Table.from_pylist(kept), OUT / "scored_pairs.parquet")
    margins = [k["positive_score"] - max(k["negative_scores"]) for k in kept]
    info = {"pairs_in": len(proposals), "pairs_kept": len(kept), **stats, "reranker": RERANKER,
            "mean_positive_logit": round(float(np.mean([k["positive_score"] for k in kept])), 3),
            "mean_hardest_negative_logit": round(float(np.mean([max(k["negative_scores"]) for k in kept])), 3),
            "median_logit_margin": round(float(np.median(margins)), 3)}  # fmt: skip
    return info


def main() -> None:
    started = time.monotonic()
    print(f"{torch.cuda.device_count()} GPU(s) | torch {torch.__version__}", flush=True)
    propose()
    gpus = torch.cuda.device_count()
    shards = max(1, gpus)
    procs = []
    for shard in range(shards):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(shard)} if gpus else dict(os.environ)
        procs.append(subprocess.Popen([sys.executable, __file__, "--rerank", str(shard), str(shards)], env=env))
    codes = [p.wait() for p in procs]
    if any(codes):
        sys.exit(f"❌ reranker process failed: exit codes {codes}")
    info = select(shards)
    info.update({"gpus": gpus, "minutes_total": round((time.monotonic() - started) / 60, 1)})
    (OUT / "teacher_pairs_info.json").write_text(json.dumps(info, indent=2))
    print(f"🏁 {json.dumps(info)}", flush=True)


if __name__ == "__main__":
    if "--rerank" in sys.argv:
        i = sys.argv.index("--rerank")
        rerank_shard(int(sys.argv[i + 1]), int(sys.argv[i + 2]))
    else:
        main()
