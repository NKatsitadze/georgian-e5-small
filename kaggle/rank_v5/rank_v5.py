"""Kaggle GPU job: v5 = v3 recipe (1 epoch) + English translations of wiki-ka-QA questions -> the same
Georgian passages (source "wiki-ka-qa-en"); the Georgian holdout is identical to v3's. v3 = the v2 student (72.6 nDCG@10) trained further to RANK like the teacher.

Each step mixes two losses on every GPU:
1. Ranking (new): RANK_BATCH questions, each with its positive and RANK_NEGATIVES of the teacher-mined
   hard negatives. The student's score distribution over those passages (softmax of cosine /
   STUDENT_TEMPERATURE) imitates the teacher's (softmax of bge-m3 score / TEACHER_TEMPERATURE) by
   KL divergence. Soft teacher scores, not hard right/wrong labels, so an imperfect question or a
   negative that is partly relevant costs little.
2. Copying (kept from v2): DISTILL_BATCH general texts whose embeddings must stay close to the
   teacher's PCA-384 targets (1 - cosine), so the student keeps what it already learned.

All texts of a step go through ONE forward pass (one DDP graph per step). Starts from the v2
checkpoint; saves a model after each epoch plus rank_v5_info.json.
Inputs: train_texts.parquet + teacher_bge-m3.npy (as for v2), scored_pairs.parquet + passages.parquet
(teacher_pairs job), the v2 checkpoint (dataset with a student-epoch6 folder).
Push from the project root:  uv run kaggle kernels push -p kaggle/rank_v5
"""

import contextlib
import json
import math
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # before torch touches the GPU
import random
import socket
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import sentence_transformers
import torch
import torch.distributed as dist
import torch.nn.functional as F
import transformers
from sentence_transformers import SentenceTransformer
from torch.nn.parallel import DistributedDataParallel

PROMPTS = {"query": "query: ", "document": "passage: "}  # as the v2 student was trained
MAX_TOKENS = 320
EPOCHS = 1  # v3: epoch 1 (77.4) >= epoch 2 (77.2)
RANK_BATCH = 32  # questions per GPU per step
RANK_NEGATIVES = 4  # sampled from the 7 mined, so each epoch sees different negatives
DISTILL_BATCH = 32  # general texts per GPU per step, keeping the v2 skill
ENGLISH_SOURCE = "wiki-ka-qa-en"
MICRO_QUESTIONS = 8  # questions per forward pass; the step is split into chunks so it fits in T4 memory
DISTILL_WEIGHT = 1.0
TEACHER_TEMPERATURE = 0.05  # bge-m3 cosines sit around 0.4-0.7; this spreads them into a clear ranking
STUDENT_TEMPERATURE = 0.05
LEARNING_RATE = 2e-5  # lower than v2's 5e-5: refining a trained model, not training a new one
WARMUP_SHARE = 0.05
WEIGHT_DECAY = 0.01
PCA_FIT_SAMPLE = 100_000
HOLDOUT_QUERIES = 2_000
LOG_EVERY = 100
SEED = 42

BACKEND = os.environ.get("RV3_DEVICE", "cuda")
INPUT_ROOT = Path(os.environ.get("RV3_INPUT", "/kaggle/input"))
OUT = Path(os.environ.get("RV3_OUTPUT", "/kaggle/working"))
LIMIT = int(os.environ.get("RV3_LIMIT", "0"))  # smoke test: use only the first N pairs / texts
FORCED_WORLD = int(os.environ.get("RV3_WORLD", "0"))
SMOKE_TEXTS = 2_000  # PCA to 384 dims needs more than 384 texts, so the smoke test keeps 2,000
if LIMIT:
    HOLDOUT_QUERIES, PCA_FIT_SAMPLE, LOG_EVERY, EPOCHS, RANK_BATCH, DISTILL_BATCH, MICRO_QUESTIONS = 20, SMOKE_TEXTS, 2, 1, 4, 4, 2


# --- processes ------------------------------------------------------------------------------


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def log(rank: int, message: str) -> None:
    if rank == 0:
        print(message, flush=True)


def barrier(world: int) -> None:
    if world > 1:
        dist.barrier()


def find_input(name: str) -> Path:
    matches = sorted(INPUT_ROOT.rglob(name))
    if not matches:
        sys.exit(f"❌ {name} not found under {INPUT_ROOT}")
    return matches[0]


def find_checkpoint() -> Path:
    """The v2 epoch-6 folder (the one with modules.json). Kaggle stores an uploaded folder either
    unpacked or as the archive the CLI made (--dir-mode zip), so unpack the archive if needed."""
    matches = sorted(p.parent for p in INPUT_ROOT.rglob("modules.json") if "epoch6" in str(p.parent))
    if matches:
        return matches[0]
    archives = sorted(INPUT_ROOT.rglob("student-epoch6.zip")) + sorted(INPUT_ROOT.rglob("student-epoch6.tar"))
    if not archives:
        sys.exit(f"❌ no student-epoch6 checkpoint (folder or archive) under {INPUT_ROOT}")
    import shutil
    import tempfile

    target = Path(tempfile.gettempdir()) / "rv3_init_checkpoint"  # not /kaggle/working: kept out of the outputs
    shutil.unpack_archive(str(archives[0]), str(target))
    found = sorted(p.parent for p in target.rglob("modules.json"))
    if not found:
        sys.exit(f"❌ {archives[0].name} did not contain a sentence-transformers model")
    return found[0]


# --- targets and batches ----------------------------------------------------------------------


def pca_targets(teacher: np.ndarray, dims: int) -> torch.Tensor:
    """Same seeded PCA-384 targets as v2 (distill.py), so the copying loss keeps v2's skill."""
    rng = np.random.default_rng(SEED)
    picked = np.sort(rng.choice(len(teacher), min(PCA_FIT_SAMPLE, len(teacher)), replace=False))
    sample = torch.from_numpy(teacher[picked].astype(np.float32))
    _, _, directions = torch.linalg.svd(sample - sample.mean(dim=0), full_matrices=False)
    components = directions[:dims].T
    targets = torch.empty(len(teacher), dims)
    for start in range(0, len(teacher), 50_000):
        chunk = torch.from_numpy(teacher[start : start + 50_000].astype(np.float32)) @ components
        targets[start : start + len(chunk)] = F.normalize(chunk, dim=1)
    return targets


ALIGN_SAMPLE = 4_000


@torch.no_grad()
def align_targets(model: SentenceTransformer, texts: list[str], targets: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    """Rotate the PCA targets onto the student's current output space (orthogonal Procrustes).

    A PCA basis is only defined up to sign flips (and rotations among near-equal directions), so a
    recomputed PCA can differ from the one v2 was trained on even with the same seed: the smoke test
    started at a copy loss of ~0.7 instead of v2's ~0.12. Fitting the best rotation on a sample of
    texts makes the copying loss continue v2's skill instead of fighting it. Returns the aligned
    targets and the mean cosine to the student before and after alignment."""
    sample = random.Random(SEED).sample(range(len(texts)), min(ALIGN_SAMPLE, len(texts)))
    model.eval()
    student = model.encode([texts[i] for i in sample], batch_size=128, convert_to_tensor=True,
                           normalize_embeddings=True).float().cpu()  # fmt: skip
    model.train()
    before = float((student * targets[sample]).sum(dim=1).mean())
    u, _, vt = torch.linalg.svd(targets[sample].T @ student)
    aligned = F.normalize(targets @ (u @ vt), dim=1)
    after = float((student * aligned[sample]).sum(dim=1).mean())
    return aligned, before, after


def prepare_batch(model: SentenceTransformer, batch: list[str], device: str) -> dict:
    preprocess = getattr(model, "preprocess", None) or model.tokenize
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in preprocess(batch).items()}


@torch.no_grad()
def holdout_ranking(model: SentenceTransformer, holdout: list[dict], passages: list[str]) -> dict:
    """On held-out questions: how often the student ranks the positive first among positive + all 7
    negatives (teacher's own rate in brackets), and the mean KL to the teacher's distribution."""
    model.eval()
    queries = model.encode([PROMPTS["query"] + h["query"] for h in holdout], batch_size=128,
                           convert_to_tensor=True, normalize_embeddings=True).float().cpu()  # fmt: skip
    candidate_ids = [[h["positive_id"], *h["negative_ids"]] for h in holdout]
    unique = sorted({i for ids in candidate_ids for i in ids})
    vectors = model.encode([PROMPTS["document"] + passages[i] for i in unique], batch_size=128,
                           convert_to_tensor=True, normalize_embeddings=True).float().cpu()  # fmt: skip
    row = {pid: n for n, pid in enumerate(unique)}
    top1, teacher_top1, kls = [], [], []
    for h, q, ids in zip(holdout, queries, candidate_ids, strict=True):
        student = torch.stack([vectors[row[i]] for i in ids]) @ q
        teacher = torch.tensor([h["positive_score"], *h["negative_scores"]])
        top1.append(int(student.argmax()) == 0)
        teacher_top1.append(int(teacher.argmax()) == 0)
        kls.append(float(F.kl_div(F.log_softmax(student / STUDENT_TEMPERATURE, 0),
                                  F.softmax(teacher / TEACHER_TEMPERATURE, 0), reduction="sum")))  # fmt: skip
    model.train()
    return {"top1": round(float(np.mean(top1)), 4), "teacher_top1": round(float(np.mean(teacher_top1)), 4),
            "kl": round(float(np.mean(kls)), 4)}  # fmt: skip


# --- training ----------------------------------------------------------------------------------


def train(rank: int, world: int, device: str) -> None:
    started = time.monotonic()
    random.seed(SEED + rank)
    torch.manual_seed(SEED)
    device_name = torch.cuda.get_device_name(device) if BACKEND == "cuda" else "cpu"
    log(rank, f"{world} x {device_name} | torch {torch.__version__} | transformers {transformers.__version__}"
              f" | sentence-transformers {sentence_transformers.__version__}")  # fmt: skip

    pairs = pq.read_table(find_input("scored_pairs.parquet")).to_pylist()
    passages = pq.read_table(find_input("passages.parquet")).column("text").to_pylist()
    table = pq.read_table(find_input("train_texts.parquet"), columns=["text", "kind"])
    texts = [PROMPTS["query" if k == "question" else "document"] + t
             for t, k in zip(table.column("text").to_pylist(), table.column("kind").to_pylist(), strict=True)]  # fmt: skip
    teacher = np.load(find_input("teacher_bge-m3.npy"), mmap_mode="r")
    georgian = [p for p in pairs if p["source"] != ENGLISH_SOURCE]  # identical to v3's pairs, same order
    english = [p for p in pairs if p["source"] == ENGLISH_SOURCE]
    if LIMIT:
        georgian, english, texts, teacher = georgian[:LIMIT], english[: LIMIT // 4], texts[:SMOKE_TEXTS], teacher[:SMOKE_TEXTS]

    model = SentenceTransformer(os.environ["RV3_CHECKPOINT"], device=device)  # resolved once in main()
    model.max_seq_length = MAX_TOKENS
    targets = pca_targets(teacher, model.get_sentence_embedding_dimension())
    targets, cosine_raw, cosine_aligned = align_targets(model, texts, targets)  # same result in every process
    log(rank, f"🎯 copy targets: student cosine {cosine_raw:.3f} before alignment, {cosine_aligned:.3f} after")
    order = list(range(len(georgian)))
    random.Random(SEED).shuffle(order)  # same split in every process, and the same Georgian holdout as v3
    holdout = [georgian[i] for i in order[:HOLDOUT_QUERIES]]
    held_passages = {h["positive_id"] for h in holdout}
    english = [e for e in english if e["positive_id"] not in held_passages]  # no translation of a held-out question
    random.Random(SEED + 1).shuffle(english)
    holdout_en = english[: HOLDOUT_QUERIES // 4]  # English holdout: 500 translated questions never trained on
    train_pairs = [georgian[i] for i in order[HOLDOUT_QUERIES:]] + english[HOLDOUT_QUERIES // 4 :]
    log(rank, f"📄 {len(train_pairs)} training pairs ({len(english) - len(holdout_en)} English), {len(holdout)} + {len(holdout_en)}"
              f" held out, {len(passages)} passages, {len(texts)} copy texts")  # fmt: skip
    before = holdout_ranking(model, holdout, passages) if rank == 0 else {}
    before_en = holdout_ranking(model, holdout_en, passages) if rank == 0 and holdout_en else {}
    log(rank, f"📏 holdout before: {before} | English: {before_en}")
    barrier(world)

    trained = DistributedDataParallel(model, find_unused_parameters=True) if world > 1 else model
    if world > 1 and BACKEND == "cuda":
        from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

        trained.register_comm_hook(state=None, hook=default_hooks.fp16_compress_hook)
    steps_per_epoch = len(train_pairs) // (RANK_BATCH * world)
    total_steps = steps_per_epoch * EPOCHS
    warmup = int(total_steps * WARMUP_SHARE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: s / max(1, warmup) if s < warmup else max(0.0, (total_steps - s) / max(1, total_steps - warmup))
    )  # fmt: skip
    use_fp16 = BACKEND == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    text_rng = random.Random(SEED * 1000 + rank)
    log(rank, f"🏋️ {world} process(es) | {RANK_BATCH} questions x (1+{RANK_NEGATIVES}) + {DISTILL_BATCH} copy texts per GPU"
              f" | {steps_per_epoch} steps/epoch x {EPOCHS}")  # fmt: skip

    history, step = [], 0
    for epoch in range(1, EPOCHS + 1):
        epoch_order = list(range(len(train_pairs)))
        random.Random(SEED + epoch).shuffle(epoch_order)  # identical in every process
        sums, epoch_started = {"rank": 0.0, "copy": 0.0}, time.monotonic()
        for step_in_epoch in range(steps_per_epoch):
            start = (step_in_epoch * world + rank) * RANK_BATCH
            batch = [train_pairs[i] for i in epoch_order[start : start + RANK_BATCH]]
            queries = [PROMPTS["query"] + b["query"] for b in batch]
            picked = [text_rng.sample(range(len(b["negative_ids"])), RANK_NEGATIVES) for b in batch]
            candidates = [PROMPTS["document"] + passages[pid] for b, picks in zip(batch, picked, strict=True)
                          for pid in [b["positive_id"], *(b["negative_ids"][k] for k in picks)]]  # fmt: skip
            copy_ids = text_rng.sample(range(len(texts)), DISTILL_BATCH)
            teacher_all = torch.tensor(
                [[b["positive_score"], *(b["negative_scores"][k] for k in picks)] for b, picks in zip(batch, picked, strict=True)],
                device=device,
            ) / TEACHER_TEMPERATURE  # fmt: skip

            # Each question's loss depends only on its own candidates and each copy text's loss only on itself,
            # so splitting the step into chunks and summing the gradients is exact (same result, less memory).
            optimizer.zero_grad(set_to_none=True)
            chunks = math.ceil(len(batch) / MICRO_QUESTIONS)
            rank_loss, copy_loss = 0.0, 0.0
            for chunk in range(chunks):
                qs = slice(chunk * MICRO_QUESTIONS, (chunk + 1) * MICRO_QUESTIONS)
                cs = slice(qs.start * (1 + RANK_NEGATIVES), qs.stop * (1 + RANK_NEGATIVES))
                chunk_copy = copy_ids[chunk::chunks]
                chunk_queries, chunk_candidates = queries[qs], candidates[cs]
                features = prepare_batch(model, chunk_queries + chunk_candidates + [texts[i] for i in chunk_copy], device)
                last = chunk == chunks - 1
                sync = contextlib.nullcontext() if last or world == 1 else trained.no_sync()
                with sync:
                    with torch.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
                        embeddings = trained(features)["sentence_embedding"]
                    embeddings = F.normalize(embeddings.float(), dim=1)
                    n_q, n_c = len(chunk_queries), len(chunk_candidates)
                    q = embeddings[:n_q]
                    c = embeddings[n_q : n_q + n_c].view(n_q, 1 + RANK_NEGATIVES, -1)
                    copied = embeddings[n_q + n_c :]
                    student_scores = torch.einsum("bd,bkd->bk", q, c) / STUDENT_TEMPERATURE
                    chunk_rank = F.kl_div(F.log_softmax(student_scores, 1), F.softmax(teacher_all[qs], 1), reduction="sum") / len(batch)
                    chunk_copy_loss = (1 - (copied * targets[chunk_copy].to(device)).sum(dim=1)).sum() / len(copy_ids)
                    scaler.scale(chunk_rank + DISTILL_WEIGHT * chunk_copy_loss).backward()
                rank_loss += chunk_rank.item()
                copy_loss += chunk_copy_loss.item()
            loss = torch.tensor(rank_loss + DISTILL_WEIGHT * copy_loss)
            rank_loss, copy_loss = torch.tensor(rank_loss), torch.tensor(copy_loss)

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            schedule.step()
            step += 1
            if not math.isfinite(loss.item()):
                sys.exit(f"❌ loss became {loss.item()} at step {step}")
            sums["rank"] += rank_loss.item()
            sums["copy"] += copy_loss.item()
            if step % LOG_EVERY == 0:
                rate = (step_in_epoch + 1) / (time.monotonic() - epoch_started)
                log(rank, f"epoch {epoch} step {step}/{total_steps} | rank KL {sums['rank'] / LOG_EVERY:.4f}"
                          f" | copy {sums['copy'] / LOG_EVERY:.4f} | {rate:.2f} steps/s"
                          f" | ~{(total_steps - step) / rate / 60:.0f} min left")  # fmt: skip
                sums = {"rank": 0.0, "copy": 0.0}

        barrier(world)
        if rank == 0:
            result = holdout_ranking(model, holdout, passages)
            result["english"] = holdout_ranking(model, holdout_en, passages) if holdout_en else {}
            model.prompts = PROMPTS
            model.save(str(OUT / f"student-v5-epoch{epoch}"))
            history.append({"epoch": epoch, **result})
            log(rank, f"✅ epoch {epoch}: holdout {result} | saved student-v5-epoch{epoch}")
        barrier(world)

    if rank == 0:
        info = {
            "init": "distill-v2/student-epoch6", "teacher": "BAAI/bge-m3", "processes": world, "device": device_name,
            "train_pairs": len(train_pairs), "holdout_before": before, "holdout_before_english": before_en, "epochs": history,
            "settings": {"rank_batch": RANK_BATCH, "rank_negatives": RANK_NEGATIVES, "distill_batch": DISTILL_BATCH, "micro_questions": MICRO_QUESTIONS,
                         "distill_weight": DISTILL_WEIGHT, "teacher_temperature": TEACHER_TEMPERATURE,
                         "student_temperature": STUDENT_TEMPERATURE, "learning_rate": LEARNING_RATE,
                         "epochs": EPOCHS, "max_tokens": MAX_TOKENS, "seed": SEED},
            "minutes_total": round((time.monotonic() - started) / 60, 1),
        }  # fmt: skip
        (OUT / "rank_v5_info.json").write_text(json.dumps(info, indent=2))
        log(rank, f"🏁 {json.dumps(info)}")


def process_main(rank: int, world: int) -> None:
    if BACKEND == "cuda":
        torch.cuda.set_device(rank)
        dist.init_process_group("nccl", rank=rank, world_size=world, timeout=timedelta(minutes=30))
        device = f"cuda:{rank}"
    else:
        dist.init_process_group("gloo", rank=rank, world_size=world, timeout=timedelta(minutes=30))
        device = "cpu"
    try:
        train(rank, world, device)
    finally:
        dist.destroy_process_group()


def main() -> int:
    if BACKEND == "cuda" and not torch.cuda.is_available():
        sys.exit("❌ no GPU: enable it for this kernel")
    world = FORCED_WORLD or (torch.cuda.device_count() if BACKEND == "cuda" else 1)
    os.environ["RV3_CHECKPOINT"] = str(find_checkpoint())  # unpack once, before the processes start
    print(f"🧩 starting from {os.environ['RV3_CHECKPOINT']}", flush=True)
    if world > 1:
        os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "127.0.0.1", str(free_port())
        print(f"🖥️ starting {world} training processes", flush=True)
        torch.multiprocessing.spawn(process_main, args=(world,), nprocs=world, join=True)
        return 0
    train(0, 1, BACKEND)
    return 0


if __name__ == "__main__":
    sys.exit(main())
