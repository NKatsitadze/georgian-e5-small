"""Build the student's ceiling model: the teacher (bge-m3) followed by the same PCA compression
(1024 -> 384) the distillation targets used. Benchmarking it shows how much search quality the
compression alone costs, i.e. the best score a perfect student could reach.

    uv run python pca_ceiling.py           # writes data/models/ceiling/bge-m3-pca384
    uv run python baseline.py --models data/models/ceiling/bge-m3-pca384

Needs the teacher kernel's output in data/teacher/ (kaggle kernels output ... -p data/teacher).
"""

import sys
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Dense, Normalize

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / "kaggle" / "distill"))
from distill import pca_components  # noqa: E402  (same seed and sample as the training job)

TEACHER_VECTORS = PROJECT / "data" / "teacher" / "teacher_bge-m3.npy"
OUT = PROJECT / "data" / "models" / "ceiling" / "bge-m3-pca384"
DIMS = 384


def main() -> None:
    teacher_vectors = np.load(TEACHER_VECTORS, mmap_mode="r")
    components, variance_kept = pca_components(teacher_vectors, DIMS)
    print(f"PCA {teacher_vectors.shape[1]} -> {DIMS}: keeps {variance_kept:.1%} of the variance")

    teacher = SentenceTransformer("BAAI/bge-m3", device="cpu")
    projection = Dense(teacher_vectors.shape[1], DIMS, bias=False, activation_function=torch.nn.Identity())
    projection.linear.weight.data = components.T.contiguous()  # nn.Linear stores [out, in]
    ceiling = SentenceTransformer(modules=[*teacher, projection, Normalize()], device="cpu")
    ceiling.save(str(OUT))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
