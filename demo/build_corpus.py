"""Build the demo's search collection: opening paragraphs of the 30,000 longest Georgian Wikipedia articles,
with vectors precomputed for both models so the Space starts instantly.

    uv run python demo/build_corpus.py

Output (demo/space/): corpus.parquet (title, url, text), emb_ours.npy, emb_base.npy (float16).
Wikipedia text is CC BY-SA 4.0; every result in the demo links to its article.
"""

import glob
import re
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "demo" / "space"
ARTICLES = 30_000
MIN_CHARS, MAX_CHARS = 150, 900
OURS, BASE = "nkatsitadze/georgian-e5-small", "intfloat/multilingual-e5-small"


def is_markup(line: str) -> bool:
    """Leftover wiki markup in the dump, e.g. infobox lines '{{ინფოდაფა ...', '|თარიღი=...', '}}'."""
    return line.startswith(("{", "|", "}", "!", "[[", "<")) or (line.count("=") >= 1 and line.startswith(("|", " |")))


def opening(text: str) -> str:
    """First real paragraphs (skipping leftover infobox markup) up to MAX_CHARS, cut at a sentence end."""
    lines = [p.strip() for p in text.split("\n") if p.strip() and not is_markup(p.strip())]
    # Infobox values survive as short lines ("თურქი სელჯუკები", "55 600"): keep only real prose paragraphs.
    paragraphs = [p for p in lines if len(p) >= 60 and len(p.split()) >= 8]
    out = ""
    for paragraph in paragraphs:
        if len(out) + len(paragraph) > MAX_CHARS:
            break
        out = f"{out} {paragraph}".strip()
    if not out and paragraphs:
        first = paragraphs[0][:MAX_CHARS]
        cut = max(first.rfind(". "), first.rfind("."))
        out = first[: cut + 1] if cut > MIN_CHARS else first
    return re.sub(r"\s+", " ", out)


def main() -> None:
    files = sorted(glob.glob(str(Path.home() / ".cache/huggingface/hub/datasets--wikimedia--wikipedia/snapshots/*/20231101.ka/*.parquet")))
    rows = [r for f in files for r in pq.read_table(f, columns=["title", "url", "text"]).to_pylist()]
    rows.sort(key=lambda r: -len(r["text"]))  # long articles = substantial topics, not one-line village stubs
    corpus = []
    for r in rows:
        text = opening(r["text"])
        # Wikipedia openings are full of dates and brackets, so only require a real paragraph that starts with a letter.
        if len(text) >= MIN_CHARS and text[0].isalpha():
            corpus.append({"title": r["title"], "url": r["url"], "text": text})
        if len(corpus) == ARTICLES:
            break
    print(f"📄 {len(corpus)} passages from {len(rows)} articles", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(corpus), OUT / "corpus.parquet")
    texts = [c["text"] for c in corpus]
    ours = SentenceTransformer(OURS, device="cpu").encode(texts, prompt_name="document", normalize_embeddings=True,
                                                          batch_size=64, show_progress_bar=True)  # fmt: skip
    np.save(OUT / "emb_ours.npy", ours.astype(np.float16))
    # The original model scores best on Georgian WITHOUT its e5 prefixes (61.5 vs 57.3), so it gets its best setting.
    base = SentenceTransformer(BASE, device="cpu").encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
    np.save(OUT / "emb_base.npy", base.astype(np.float16))
    print(f"✅ saved to {OUT}", flush=True)


if __name__ == "__main__":
    main()
