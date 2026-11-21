"""Run distill.py locally on 2,000 texts with random stand-in teacher vectors, on the CPU.

It only checks the plumbing (loading, PCA, training loop, saving), not quality, so a bug shows
up in about a minute here instead of after a GPU run on Kaggle:
    uv run python kaggle/distill/smoke_test.py
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

PROJECT = Path(__file__).resolve().parents[2]
TEXTS = 2_000
TEACHER_DIMS = 1024


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        inputs, outputs = Path(tmp, "input"), Path(tmp, "working")
        inputs.mkdir()
        outputs.mkdir()
        table = pq.read_table(PROJECT / "data" / "train_texts.parquet").slice(0, TEXTS)
        pq.write_table(table, inputs / "train_texts.parquet")
        fake = np.random.default_rng(0).standard_normal((TEXTS, TEACHER_DIMS)).astype(np.float32)
        np.save(inputs / "teacher_bge-m3.npy", (fake / np.linalg.norm(fake, axis=1, keepdims=True)).astype(np.float16))

        # DISTILL_WORLD=2: two CPU processes stand in for Kaggle's two GPUs, so the DDP path is tested.
        env = {**os.environ, "DISTILL_DEVICE": "cpu", "DISTILL_INPUT": str(inputs),
               "DISTILL_OUTPUT": str(outputs), "DISTILL_LIMIT": str(TEXTS), "DISTILL_WORLD": "2"}  # fmt: skip
        try:
            result = subprocess.run([sys.executable, str(Path(__file__).with_name("distill.py"))], env=env, timeout=900)
        except subprocess.TimeoutExpired:
            print("❌ smoke test timed out after 15 min (hung process group?)")
            return 1
        saved = sorted(p.name for p in outputs.iterdir())
        print(f"outputs: {saved}")
        ok = result.returncode == 0 and "student-epoch1" in saved and "training_info.json" in saved
        print("✅ smoke test passed" if ok else "❌ smoke test failed")
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
