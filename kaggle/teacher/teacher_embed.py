"""Kaggle GPU job 1: the teacher (BAAI/bge-m3) embeds every training text once.

Output (in /kaggle/working, kept as this kernel's output):
- teacher_bge-m3.npy: float16 array [n_texts, 1024], L2-normalized, same row order as the parquet
- teacher_info.json: counts, timings and library versions

The training job reads this output, so the student can be retrained with other settings without
paying for the teacher again. Push from the project root:
    uv run kaggle kernels push -p kaggle/teacher
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import sentence_transformers
import torch
import transformers
from sentence_transformers import SentenceTransformer

TEACHER = "BAAI/bge-m3"
MAX_TOKENS = 320  # covers ~all passages (median ~220 Georgian tokens); the student uses the same cap
BATCH_SIZE = 128
OUT = Path("/kaggle/working")


def find_input(name: str) -> Path:
    """Kaggle mounts datasets under /kaggle/input/, but the exact sub-path varies."""
    matches = sorted(Path("/kaggle/input").rglob(name))
    if not matches:
        sys.exit(f"❌ {name} not found under /kaggle/input")
    return matches[0]


def encode_on_all_gpus(model: SentenceTransformer, texts: list[str], devices: list[str]) -> np.ndarray:
    """Use every GPU Kaggle gives us (e.g. T4 x2): one worker process per GPU, each embedding a
    share of the texts. Falls back to plain single-GPU encoding when there is only one."""
    options = {"batch_size": BATCH_SIZE, "normalize_embeddings": True}
    if len(devices) == 1:
        return model.encode(texts, convert_to_numpy=True, show_progress_bar=True, **options)
    pool = model.start_multi_process_pool(target_devices=devices)
    try:
        try:
            return model.encode(texts, pool=pool, convert_to_numpy=True, show_progress_bar=True, **options)
        except TypeError:  # older sentence-transformers: encode() has no pool argument
            return model.encode_multi_process(texts, pool, **options)
    finally:
        model.stop_multi_process_pool(pool)


def main() -> None:
    started = time.monotonic()
    if not torch.cuda.is_available():
        sys.exit("❌ no GPU: enable it for this kernel (Kaggle needs a verified phone number for GPUs)")
    gpu = torch.cuda.get_device_name(0)
    print(f"GPU {gpu} | torch {torch.__version__} | transformers {transformers.__version__}"
          f" | sentence-transformers {sentence_transformers.__version__}", flush=True)  # fmt: skip

    texts = pq.read_table(find_input("train_texts.parquet"), columns=["text"]).column("text").to_pylist()
    print(f"📄 {len(texts)} texts", flush=True)

    model = SentenceTransformer(TEACHER, device="cuda")
    model.half()  # fp16: about 2x faster on the GPU, embeddings are stored as float16 anyway
    model.max_seq_length = MAX_TOKENS

    devices = [f"cuda:{index}" for index in range(torch.cuda.device_count())]
    print(f"🖥️ encoding on {len(devices)} GPU(s): {devices}", flush=True)
    encode_started = time.monotonic()
    embeddings = encode_on_all_gpus(model, texts, devices)
    encode_seconds = time.monotonic() - encode_started

    if not np.isfinite(embeddings).all():
        sys.exit("❌ non-finite values in the embeddings (fp16 overflow?); nothing saved")
    np.save(OUT / "teacher_bge-m3.npy", embeddings.astype(np.float16))
    info = {
        "teacher": TEACHER,
        "texts": len(texts),
        "shape": list(embeddings.shape),
        "max_tokens": MAX_TOKENS,
        "texts_per_second": round(len(texts) / encode_seconds, 1),
        "minutes_total": round((time.monotonic() - started) / 60, 1),
        "gpu": gpu,
        "gpus_used": len(devices),
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "sentence_transformers": sentence_transformers.__version__,
        },
    }
    (OUT / "teacher_info.json").write_text(json.dumps(info, indent=2))
    print(f"✅ saved {embeddings.shape} | {json.dumps(info)}", flush=True)


if __name__ == "__main__":
    main()
