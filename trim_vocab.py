"""Build the "lite" release: the full model with its vocabulary trimmed to Georgian, English and Russian.

    uv run python trim_vocab.py

Why it is safe: the tokenizer is a Unigram model that picks the best segmentation from its vocabulary.
If a text's best segmentation under the FULL vocabulary uses only kept pieces, the trimmed tokenizer
finds exactly the same segmentation, so the model sees the same tokens and returns the same vector.
We keep every piece used in a large sample of Georgian, English and Russian text, plus every
single-character piece (so unseen text degrades to characters instead of <unk>), and then measure on
HELD-OUT text how often the segmentation is identical.

Output: data/release/lite (same files as data/release/full, smaller embedding matrix and tokenizer).
"""

import json
import random
import shutil
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer

PROJECT = Path(__file__).resolve().parent
FULL = PROJECT / "data" / "release" / "full"
LITE = PROJECT / "data" / "release" / "lite"
SEED = 42
PER_SOURCE = 120_000  # texts sampled per source; 80% choose the vocabulary, 20% are held out to test it
PREFIXES = ("query: ", "passage: ")


def wiki_paragraphs(config: str, shard: str, limit: int, rng: random.Random) -> list[str]:
    path = hf_hub_download("wikimedia/wikipedia", f"{config}/{shard}", repo_type="dataset")
    texts = pq.read_table(path, columns=["text"]).column("text").to_pylist()
    paragraphs = [p.strip() for t in rng.sample(texts, min(len(texts), limit // 2)) for p in t.split("\n") if len(p.strip()) > 40]
    return rng.sample(paragraphs, min(limit, len(paragraphs)))


def sources(rng: random.Random) -> dict[str, list[str]]:
    train = pq.read_table(PROJECT / "data" / "train_texts.parquet", columns=["text"]).column("text").to_pylist()
    fineweb = hf_hub_download("HuggingFaceFW/fineweb-2", "data/kat_Geor/train/001_00000.parquet", repo_type="dataset")
    web = [t[:2000] for t in pq.read_table(fineweb, columns=["text"]).column("text").to_pylist()]
    return {
        "training texts (ka + en)": rng.sample(train, min(PER_SOURCE, len(train))),
        "Georgian web (FineWeb-2)": rng.sample(web, min(PER_SOURCE, len(web))),
        "English Wikipedia": wiki_paragraphs("20231101.en", "train-00001-of-00041.parquet", PER_SOURCE, rng),
        "Russian Wikipedia": wiki_paragraphs("20231101.ru", "train-00002-of-00021.parquet", PER_SOURCE, rng),
    }


def encode_ids(tokenizer: Tokenizer, texts: list[str]) -> list[list[int]]:
    return [e.ids for e in tokenizer.encode_batch([p + t for t in texts for p in PREFIXES])]


def main() -> None:
    rng = random.Random(SEED)
    full_tok = Tokenizer.from_file(str(FULL / "tokenizer.json"))
    full_tok.no_truncation()
    full_tok.no_padding()  # padding to the longest text in a batch would make one differing text spoil the whole batch
    spec = json.loads((FULL / "tokenizer.json").read_text(encoding="utf-8"))
    vocab = spec["model"]["vocab"]  # [[piece, score], ...]; the index is the token id

    data = {name: texts for name, texts in sources(rng).items()}
    split = {name: (texts[: int(len(texts) * 0.8)], texts[int(len(texts) * 0.8) :]) for name, texts in data.items()}
    counts = Counter()
    for name, (select, _) in split.items():
        for ids in encode_ids(full_tok, select):
            counts.update(ids)
        print(f"   counted {name}: {len(select)} texts, {len(counts)} distinct pieces so far", flush=True)

    special = {a["id"] for a in spec["added_tokens"]}
    single_chars = {i for i, (piece, _) in enumerate(vocab) if len(piece.replace("▁", "")) == 1}
    kept = sorted(set(counts) | special | single_chars)
    remap = {old: new for new, old in enumerate(kept)}
    print(f"✅ keeping {len(kept)} of {len(vocab)} pieces ({len(set(counts))} used, {len(single_chars)} single characters, "
          f"{len(special)} special)", flush=True)  # fmt: skip

    # Tokenizer: same pieces and scores in the same order, renumbered.
    spec["model"]["vocab"] = [vocab[i] for i in kept]
    spec["model"]["unk_id"] = remap[spec["model"]["unk_id"]]
    for added in spec["added_tokens"]:
        added["id"] = remap[added["id"]]
    for token in spec["post_processor"].get("special_tokens", {}).values():
        token["ids"] = [remap[i] for i in token["ids"]]
    if LITE.exists():
        shutil.rmtree(LITE)
    shutil.copytree(FULL, LITE)
    (LITE / "tokenizer.json").write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")

    # Model: keep only the embedding rows of the kept pieces.
    weights = load_file(str(FULL / "model.safetensors"))
    weights["embeddings.word_embeddings.weight"] = weights["embeddings.word_embeddings.weight"][torch.tensor(kept)].contiguous()
    save_file(weights, str(LITE / "model.safetensors"), metadata={"format": "pt"})
    config = json.loads((LITE / "config.json").read_text())
    config["vocab_size"] = len(kept)
    for key in ("pad_token_id", "bos_token_id", "eos_token_id"):
        if config.get(key) is not None:
            config[key] = remap[config[key]]
    (LITE / "config.json").write_text(json.dumps(config, indent=2))

    # Check on HELD-OUT text: identical segmentation (mapped to new ids) means an identical vector.
    lite_tok = Tokenizer.from_file(str(LITE / "tokenizer.json"))
    lite_tok.no_truncation()
    lite_tok.no_padding()
    report = {"kept_pieces": len(kept), "full_pieces": len(vocab)}
    for name, (_, held_out) in split.items():
        full_ids, lite_ids = encode_ids(full_tok, held_out), encode_ids(lite_tok, held_out)
        same = sum([remap.get(i, -1) for i in a] == b for a, b in zip(full_ids, lite_ids, strict=True))
        unk = sum(remap[spec_unk] in b for b in lite_ids for spec_unk in [3])
        report[name] = {"held_out_texts": len(full_ids), "identical": round(same / len(full_ids), 5), "with_unk": unk}
        print(f"   {name}: {same / len(full_ids):.3%} of held-out texts tokenized identically", flush=True)
    (LITE / "trim_report.json").write_text(json.dumps(report, indent=2))
    size = lambda p: (p / "model.safetensors").stat().st_size / 1e6  # noqa: E731
    print(f"🏁 model.safetensors: full {size(FULL):.0f} MB -> lite {size(LITE):.0f} MB", flush=True)


if __name__ == "__main__":
    main()
