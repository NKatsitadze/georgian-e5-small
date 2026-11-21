"""Kaggle GPU job 2: distill the teacher (bge-m3) into a small student (multilingual-e5-small).

The student learns to output, for each training text, the teacher's embedding compressed from 1024
to 384 numbers with PCA (the student's own size), re-normalized to unit length. Unit-length targets
make the MSE loss equivalent to maximizing cosine similarity, which is what search uses.

Uses every GPU it is given: with N > 1 GPUs it starts N processes (PyTorch DDP), each training on
its own share of every batch, with gradients averaged after each step.

Inputs: train_texts.parquet (dataset) and teacher_bge-m3.npy (output of the teacher kernel).
Output (/kaggle/working): one saved model per epoch plus training_info.json.
Push from the project root (after the teacher kernel has finished):
    uv run kaggle kernels push -p kaggle/distill
"""

import json
import math
import os
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
import transformers
from sentence_transformers import SentenceTransformer
from torch.nn.parallel import DistributedDataParallel

STUDENT = "intfloat/multilingual-e5-small"
PROMPTS = {"query": "query: ", "document": "passage: "}  # e5's prefixes; saved with the model
MAX_TOKENS = 320  # same cap the teacher used
EPOCHS = 6  # v2: only change from v1 (2 epochs); a model is saved after each epoch -> learning curve
BATCH_PER_DEVICE = 64  # global batch = 64 x number of GPUs
LEARNING_RATE = 5e-5
WARMUP_SHARE = 0.05
WEIGHT_DECAY = 0.01
HOLDOUT = 5_000  # texts never trained on, to check the student is learning
PCA_FIT_SAMPLE = 100_000
LOG_EVERY = 200
SEED = 42

# Speed switches (v3). Meant to change speed, not what is learned; the next run checks that epoch-1
# holdout cosine matches v2 while steps/s goes up. Set to False to get v2's exact behavior.
LENGTH_GROUPED = True  # batch similar-token-count texts together: v2 batches were 38% padding, this 0.6%
LENGTH_GROUP_BATCHES = 50  # sort within windows of 50 batches, then shuffle the batches
FP16_GRAD_SYNC = True  # send gradients between GPUs in fp16: halves ~470 MB of traffic per step

# Kaggle defaults; overridden only by the local smoke test (smoke_test.py).
BACKEND = os.environ.get("DISTILL_DEVICE", "cuda")  # "cuda" on Kaggle, "cpu" in the smoke test
INPUT_ROOT = Path(os.environ.get("DISTILL_INPUT", "/kaggle/input"))
OUT = Path(os.environ.get("DISTILL_OUTPUT", "/kaggle/working"))
LIMIT = int(os.environ.get("DISTILL_LIMIT", "0"))  # >0: use only the first N texts
FORCED_WORLD = int(os.environ.get("DISTILL_WORLD", "0"))  # smoke test: simulate N processes on CPU
if LIMIT:
    HOLDOUT, PCA_FIT_SAMPLE, LOG_EVERY, EPOCHS = 200, LIMIT, 5, 1


# --- processes and devices -----------------------------------------------------------------


def wanted_processes() -> int:
    if FORCED_WORLD:
        return FORCED_WORLD
    return torch.cuda.device_count() if BACKEND == "cuda" else 1


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def run_distributed(processes: int) -> int:
    """One process per GPU, meeting on 127.0.0.1. Not torchrun: its hostname-based rendezvous
    timed out on macOS, and an explicit loopback address works the same everywhere."""
    os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "127.0.0.1", str(free_port())
    print(f"🖥️ starting {processes} training processes", flush=True)
    torch.multiprocessing.spawn(process_main, args=(processes,), nprocs=processes, join=True)
    return 0


def setup_process(rank: int, world: int) -> str:
    """Join the process group and return this process's device."""
    if BACKEND == "cuda":
        torch.cuda.set_device(rank)
        dist.init_process_group("nccl", rank=rank, world_size=world, timeout=timedelta(minutes=30))
        return f"cuda:{rank}"
    dist.init_process_group("gloo", rank=rank, world_size=world, timeout=timedelta(minutes=30))
    return "cpu"


def process_main(rank: int, world: int) -> None:
    device = setup_process(rank, world)
    try:
        train(rank, world, device)
    finally:
        dist.destroy_process_group()


def log(rank: int, message: str) -> None:
    if rank == 0:
        print(message, flush=True)


def barrier(world: int) -> None:
    if world > 1:
        dist.barrier()


# --- data and targets ----------------------------------------------------------------------


def find_input(name: str) -> Path:
    matches = sorted(INPUT_ROOT.rglob(name))
    if not matches:
        sys.exit(f"❌ {name} not found under {INPUT_ROOT}")
    return matches[0]


def pca_components(teacher: np.ndarray, dims: int) -> tuple[torch.Tensor, float]:
    """Top `dims` principal directions of the teacher's embeddings, fitted on a seeded sample so
    every process (and pca_ceiling.py) gets the same result. Returns ([1024, dims], variance kept)."""
    rng = np.random.default_rng(SEED)
    picked = np.sort(rng.choice(len(teacher), min(PCA_FIT_SAMPLE, len(teacher)), replace=False))
    sample = torch.from_numpy(teacher[picked].astype(np.float32))
    _, singular, directions = torch.linalg.svd(sample - sample.mean(dim=0), full_matrices=False)
    variance_kept = float((singular[:dims] ** 2).sum() / (singular**2).sum())
    return directions[:dims].T, variance_kept


def pca_targets(teacher: np.ndarray, dims: int) -> tuple[torch.Tensor, float]:
    """The teacher's embeddings projected onto pca_components (no centering), re-normalized."""
    components, variance_kept = pca_components(teacher, dims)
    targets = torch.empty(len(teacher), dims)
    for start in range(0, len(teacher), 50_000):
        chunk = torch.from_numpy(teacher[start : start + 50_000].astype(np.float32)) @ components
        targets[start : start + len(chunk)] = torch.nn.functional.normalize(chunk, dim=1)
    return targets, variance_kept


def token_lengths(model: SentenceTransformer, texts: list[str]) -> list[int]:
    """Token count of every text (~35 s for 441k on a CPU). Grouping by characters left 19% padding,
    because Georgian, English and digits turn into tokens at very different rates; tokens: 0.6%."""
    lengths: list[int] = []
    for start in range(0, len(texts), 10_000):
        encoded = model.tokenizer(texts[start : start + 10_000], truncation=True, max_length=MAX_TOKENS)
        lengths += [len(ids) for ids in encoded["input_ids"]]
    return lengths


def epoch_batches(train_ids: list[int], lengths: list[int], epoch: int, global_batch: int, steps: int) -> list[list[int]]:
    """The epoch's global batches, identical in every process (same seed).

    With LENGTH_GROUPED, texts are sorted by length inside windows of LENGTH_GROUP_BATCHES batches,
    so each batch holds similar lengths (little padding), and the batches are then shuffled so the
    model still sees lengths in random order."""
    rng = random.Random(SEED + epoch)
    ids = train_ids[:]
    rng.shuffle(ids)
    ids = ids[: steps * global_batch]
    if not LENGTH_GROUPED:
        return [ids[i : i + global_batch] for i in range(0, len(ids), global_batch)]
    window = global_batch * LENGTH_GROUP_BATCHES
    batches = []
    for start in range(0, len(ids), window):
        group = sorted(ids[start : start + window], key=lambda i: lengths[i])
        batches += [group[i : i + global_batch] for i in range(0, len(group), global_batch)]
    rng.shuffle(batches)
    return batches


def prepare_batch(model: SentenceTransformer, batch: list[str], device: str) -> dict:
    """Tokenize and move tensors to the device. sentence-transformers 6.x renamed tokenize() to
    preprocess() and adds non-tensor fields; Kaggle's preinstalled version may be older."""
    preprocess = getattr(model, "preprocess", None) or model.tokenize
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in preprocess(batch).items()}


@torch.no_grad()
def holdout_cosine(model: SentenceTransformer, texts: list[str], targets: torch.Tensor) -> float:
    """Mean cosine between the student's embeddings and the teacher targets on unseen texts."""
    model.eval()
    embeddings = model.encode(texts, batch_size=256, convert_to_tensor=True, normalize_embeddings=True)
    model.train()
    return float((embeddings.float().cpu() * targets).sum(dim=1).mean())


# --- training ------------------------------------------------------------------------------


def train(rank: int, world: int, device: str) -> None:
    started = time.monotonic()
    random.seed(SEED)
    torch.manual_seed(SEED)
    device_name = torch.cuda.get_device_name(device) if BACKEND == "cuda" else "cpu"
    log(rank, f"{world} x {device_name} | torch {torch.__version__} | transformers {transformers.__version__}"
              f" | sentence-transformers {sentence_transformers.__version__}")  # fmt: skip

    table = pq.read_table(find_input("train_texts.parquet"), columns=["text", "kind"])
    texts = [
        PROMPTS["query" if kind == "question" else "document"] + text
        for text, kind in zip(table.column("text").to_pylist(), table.column("kind").to_pylist(), strict=True)
    ]
    teacher = np.load(find_input("teacher_bge-m3.npy"), mmap_mode="r")
    if len(teacher) != len(texts):
        sys.exit(f"❌ {len(texts)} texts but {len(teacher)} teacher embeddings: wrong input versions")
    if LIMIT:
        texts, teacher = texts[:LIMIT], teacher[:LIMIT]
        log(rank, f"⚠️ smoke test: only {LIMIT} texts")

    model = SentenceTransformer(STUDENT, device=device)
    model.max_seq_length = MAX_TOKENS
    dims = model.get_sentence_embedding_dimension()
    targets, variance_kept = pca_targets(teacher, dims)
    log(rank, f"🎯 targets {tuple(targets.shape)} | PCA keeps {variance_kept:.1%} of the teacher's variance")

    order = list(range(len(texts)))
    random.Random(SEED).shuffle(order)
    holdout, train_ids = order[:HOLDOUT], order[HOLDOUT:]
    holdout_texts, holdout_targets = [texts[i] for i in holdout], targets[holdout]
    cosine_before = holdout_cosine(model, holdout_texts, holdout_targets) if rank == 0 else 0.0
    log(rank, f"📏 holdout cosine before training: {cosine_before:.4f}")
    barrier(world)

    # find_unused_parameters: the transformer's pooler layer exists but its output is never used.
    trained = DistributedDataParallel(model, find_unused_parameters=True) if world > 1 else model
    if world > 1 and BACKEND == "cuda" and FP16_GRAD_SYNC:
        from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

        trained.register_comm_hook(state=None, hook=default_hooks.fp16_compress_hook)
    lengths = token_lengths(model, texts) if LENGTH_GROUPED else []
    global_batch = BATCH_PER_DEVICE * world
    steps_per_epoch = len(train_ids) // global_batch
    total_steps = steps_per_epoch * EPOCHS
    warmup = int(total_steps * WARMUP_SHARE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: step / max(1, warmup) if step < warmup else max(0.0, (total_steps - step) / (total_steps - warmup))
    )  # fmt: skip
    use_fp16 = BACKEND == "cuda"  # mixed precision only on NVIDIA GPUs
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    log(rank, f"🏋️ {world} process(es) | global batch {global_batch} | {steps_per_epoch} steps/epoch x {EPOCHS}")

    history = []
    step = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        plan = epoch_batches(train_ids, lengths, epoch, global_batch, steps_per_epoch)
        running, epoch_started = 0.0, time.monotonic()
        real_tokens = padded_tokens = 0
        for step_in_epoch, global_ids in enumerate(plan):
            batch = global_ids[rank * BATCH_PER_DEVICE : (rank + 1) * BATCH_PER_DEVICE]  # this process's share
            features = prepare_batch(model, [texts[i] for i in batch], device)
            real_tokens += int(features["attention_mask"].sum())
            padded_tokens += features["attention_mask"].numel()
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
                embeddings = trained(features)["sentence_embedding"]
            loss = torch.nn.functional.mse_loss(embeddings.float(), targets[batch].to(device))
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            schedule.step()
            step += 1
            if not math.isfinite(loss.item()):
                sys.exit(f"❌ loss became {loss.item()} at step {step}")
            running += loss.item()
            if step % LOG_EVERY == 0:
                rate = (step_in_epoch + 1) / (time.monotonic() - epoch_started)
                left = (total_steps - step) / rate / 60
                padding = 1 - real_tokens / max(1, padded_tokens)
                log(rank, f"epoch {epoch} step {step}/{total_steps} | loss {running / LOG_EVERY:.6f} | "
                          f"{rate:.1f} steps/s | padding {padding:.0%} | ~{left:.0f} min left")  # fmt: skip
                running = 0.0
                real_tokens = padded_tokens = 0

        barrier(world)
        if rank == 0:
            cosine = holdout_cosine(model, holdout_texts, holdout_targets)
            model.prompts = PROMPTS
            model.save(str(OUT / f"student-epoch{epoch}"))
            history.append({"epoch": epoch, "holdout_cosine": round(cosine, 4)})
            log(rank, f"✅ epoch {epoch}: holdout cosine {cosine:.4f} | saved student-epoch{epoch}")
        barrier(world)

    if rank == 0:
        info = {
            "student": STUDENT,
            "teacher": "BAAI/bge-m3",
            "processes": world,
            "device": device_name,
            "train_texts": len(train_ids),
            "holdout_texts": len(holdout),
            "pca_variance_kept": round(variance_kept, 4),
            "holdout_cosine_before": round(cosine_before, 4),
            "epochs": history,
            "settings": {"max_tokens": MAX_TOKENS, "batch_per_device": BATCH_PER_DEVICE, "global_batch": global_batch,
                         "learning_rate": LEARNING_RATE, "warmup_share": WARMUP_SHARE,
                         "weight_decay": WEIGHT_DECAY, "seed": SEED, "length_grouped": LENGTH_GROUPED,
                         "fp16_grad_sync": FP16_GRAD_SYNC, "epochs": EPOCHS},  # fmt: skip
            "minutes_total": round((time.monotonic() - started) / 60, 1),
        }
        (OUT / "training_info.json").write_text(json.dumps(info, indent=2))
        log(rank, f"🏁 {json.dumps(info)}")


def main() -> int:
    if BACKEND == "cuda" and not torch.cuda.is_available():
        sys.exit("❌ no GPU: enable it for this kernel")
    processes = wanted_processes()
    if processes > 1:
        return run_distributed(processes)
    train(0, 1, BACKEND)
    return 0


if __name__ == "__main__":
    sys.exit(main())
