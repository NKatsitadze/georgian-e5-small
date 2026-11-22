"""Compact data files for the static (in-browser) demo, from demo/space/ (built by demo/build_corpus.py).

    uv run python demo/build_static.py

Output (demo/static/):
- docs.json.gz: titles and short snippets, gzipped (the Space's CDN doesn't compress large files; the browser
  unpacks it with DecompressionStream). Each result links to its Wikipedia article for the full text.
- emb_ours.i8 / emb_ours.f32, emb_base.i8 / emb_base.f32: document vectors as int8 with one float32 scale per
  vector (a quarter of the float32 size). The browser computes score = scale * (int8 vector · query vector).
"""

import gzip
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

PROJECT = Path(__file__).resolve().parents[1]
SRC, OUT = PROJECT / "demo" / "space", PROJECT / "demo" / "static"
SNIPPET_CHARS = 240


def snippet(text: str) -> str:
    return text if len(text) <= SNIPPET_CHARS else text[:SNIPPET_CHARS].rsplit(" ", 1)[0] + " …"


def quantize(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scale = np.abs(vectors).max(axis=1) / 127.0
    return np.round(vectors / scale[:, None]).astype(np.int8), scale.astype(np.float32)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    corpus = pq.read_table(SRC / "corpus.parquet").to_pylist()
    docs = {"title": [c["title"] for c in corpus], "snippet": [snippet(c["text"]) for c in corpus]}
    (OUT / "docs.json.gz").write_bytes(gzip.compress(json.dumps(docs, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), 9))
    (OUT / "docs.json").unlink(missing_ok=True)
    rng = np.random.default_rng(0)
    for key in ("ours", "base"):
        vectors = np.load(SRC / f"emb_{key}.npy").astype(np.float32)
        q, scale = quantize(vectors)
        q.tofile(OUT / f"emb_{key}.i8")
        scale.tofile(OUT / f"emb_{key}.f32")
        # Check: with random query directions, int8 scores must rank documents like the float scores.
        queries = vectors[rng.choice(len(vectors), 200, replace=False)] + rng.normal(0, 0.02, (200, vectors.shape[1]))
        exact, approx = queries @ vectors.T, (queries @ q.astype(np.float32).T) * scale
        same_top10 = np.mean([len(set(np.argsort(-e)[:10]) & set(np.argsort(-a)[:10])) / 10 for e, a in zip(exact, approx, strict=True)])
        print(f"{key}: {q.nbytes / 1e6:.1f} MB int8 | top-10 overlap with float vectors: {same_top10:.1%}")
    print(f"docs.json.gz: {(OUT / 'docs.json.gz').stat().st_size / 1e6:.1f} MB ({len(corpus)} passages)")


if __name__ == "__main__":
    main()
