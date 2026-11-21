"""Kaggle GPU job: which open (Apache-2.0) LLM translates Georgian search questions into English well?

Goal (v5): English question -> Georgian passage training pairs, our weakest benchmark direction.
Each candidate translates the same 100 wiki-ka-QA questions (greedy decoding), seeing the
passage as context (pilot 2: without it, ~1 in 4 translations got a rare verb wrong). Then the teacher
(bge-m3) checks each translation two ways:
- meaning kept: cosine between the Georgian question and its English translation;
- still answerable: rank of the question's own Georgian passage among 5,100 passages, for the
  Georgian original and for the English translation (a good translation ranks it about as well).
A human then reviews a sample, because these checks are blind to wording quality.

Output (/kaggle/working): translate_results.jsonl and translate_stats.json.
Push from the project root:  uv run kaggle kernels push -p kaggle/translate_pilot
"""

import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

# (name, Hugging Face repo, load in 4-bit). Qwen3 is Apache-2.0: its output may be used to train
# and publish a model without extra conditions.
CANDIDATES = [
    ("qwen3-8b", "Qwen/Qwen3-8B", False),  # fp16, ~16 GB, split over the two T4s
    ("qwen3-14b-4bit", "Qwen/Qwen3-14B", True),  # 4-bit NF4, ~9 GB
]
TEACHER = "BAAI/bge-m3"
PROMPT = """Translate a Georgian search question into natural English, the way an English speaker would type it into a search engine.

The passage the question is about is given only as context, so you understand rare words and names correctly.

Rules:
- Translate the question only. Keep its full meaning; do not add, drop or explain anything, and do not answer it.
- Write names of people and places in their usual English spelling.
- Return only the English question, nothing else.

Passage (context only):
{passage}

Georgian question:
{question}"""
GENERATION = {"max_new_tokens": 96, "do_sample": False}
BATCH_SIZE = 8  # prompts now include the passage

INPUT_ROOT = Path(os.environ.get("TR_INPUT", "/kaggle/input"))
OUT = Path(os.environ.get("TR_OUTPUT", "/kaggle/working"))
LIMIT = int(os.environ.get("TR_LIMIT", "0"))
if os.environ.get("TR_CANDIDATES"):  # local smoke test, e.g. "tiny=Qwen/Qwen3-0.6B"
    CANDIDATES = [(name, repo, False) for name, repo in (c.split("=") for c in os.environ["TR_CANDIDATES"].split(","))]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def find_input(name: str) -> Path:
    matches = sorted(INPUT_ROOT.rglob(name))
    if not matches:
        sys.exit(f"❌ {name} not found under {INPUT_ROOT}")
    return matches[0]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def clean(raw: str) -> str:
    """First non-empty line, without quotes or a leftover thinking block."""
    text = raw.split("</think>")[-1].strip()
    line = next((part.strip() for part in text.splitlines() if part.strip()), "")
    return line.strip(" \"'„“«»*").strip()


def translate(repo: str, four_bit: bool, pairs: list[dict]) -> tuple[list[str], dict]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(repo, padding_side="left")
    options: dict = {"dtype": torch.float16 if DEVICE == "cuda" else torch.float32}
    if DEVICE == "cuda":
        options["device_map"] = "auto"  # spreads a model that doesn't fit one GPU over both
    if four_bit:
        from transformers import BitsAndBytesConfig

        options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
        )
    model = AutoModelForCausalLM.from_pretrained(repo, **options)
    model.eval()
    prompts = [
        tokenizer.apply_chat_template([{"role": "user", "content": PROMPT.format(passage=p["passage"], question=p["query"])}],
                                      tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for p in pairs
    ]  # fmt: skip
    out, started = [], time.monotonic()
    for start in range(0, len(prompts), BATCH_SIZE):
        batch = tokenizer(prompts[start : start + BATCH_SIZE], return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            generated = model.generate(**batch, **GENERATION, pad_token_id=tokenizer.pad_token_id)
        out += [clean(t) for t in tokenizer.batch_decode(generated[:, batch["input_ids"].shape[1] :], skip_special_tokens=True)]
        print(f"   {min(start + BATCH_SIZE, len(prompts))}/{len(prompts)}", flush=True)
    seconds = time.monotonic() - started
    del model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return out, {"seconds": round(seconds, 1), "questions_per_minute": round(len(pairs) / seconds * 60, 1)}


def main() -> None:
    started = time.monotonic()
    print(f"device {DEVICE} x {torch.cuda.device_count() if DEVICE == 'cuda' else 1} | torch {torch.__version__}", flush=True)
    pairs = read_jsonl(find_input("pilot_pairs.jsonl"))
    distractors = [d["text"] for d in read_jsonl(find_input("distractors.jsonl"))]
    if LIMIT:
        pairs, distractors = pairs[:LIMIT], distractors[: LIMIT * 20]

    translations, stats = {}, {}
    for name, repo, four_bit in CANDIDATES:
        print(f"▶ {name} ({repo}{', 4-bit' if four_bit else ''})", flush=True)
        try:
            translations[name], stats[name] = translate(repo, four_bit, pairs)
        except Exception as error:  # one failing candidate must not lose the others' results
            print(f"❌ {name}: {type(error).__name__}: {error}", flush=True)
            stats[name] = {"error": f"{type(error).__name__}: {error}"}
            continue
        print(f"✅ {name}: {stats[name]}", flush=True)
    if not translations:
        sys.exit("❌ no candidate produced translations")

    from sentence_transformers import SentenceTransformer

    print(f"▶ checks with {TEACHER}", flush=True)
    teacher = SentenceTransformer(TEACHER, device=DEVICE)
    if DEVICE == "cuda":
        teacher.half()
    encode = lambda texts: teacher.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_tensor=True).float()  # noqa: E731
    pool = encode([p["passage"] for p in pairs] + distractors)

    def ranks(vectors: torch.Tensor) -> list[int]:
        scores = vectors @ pool.T
        own = scores[torch.arange(len(pairs)), torch.arange(len(pairs))]
        return ((scores > own[:, None]).sum(dim=1) + 1).tolist()

    georgian = encode([p["query"] for p in pairs])
    georgian_ranks = ranks(georgian)
    rows = [{"id": p["id"], "query_ka": p["query"], "passage": p["passage"], "rank_ka": r} for p, r in zip(pairs, georgian_ranks, strict=True)]
    for name, texts in translations.items():
        english = encode([t or "-" for t in texts])
        similarity = (english * georgian).sum(dim=1).tolist()
        english_ranks = ranks(english)
        for row, text, sim, rank in zip(rows, texts, similarity, english_ranks, strict=True):
            row[name] = {"en": text, "similarity": round(sim, 4), "rank_en": rank}
        stats[name].update({
            "mean_similarity": round(float(np.mean(similarity)), 4),
            "low_similarity_share": round(float(np.mean([s < 0.8 for s in similarity])), 3),
            "top1_en": round(float(np.mean([r == 1 for r in english_ranks])), 3),
            "empty": sum(not t for t in texts),
        })  # fmt: skip
    summary = {"models": stats, "pairs": len(pairs), "pool": len(pool),
               "top1_ka": round(float(np.mean([r == 1 for r in georgian_ranks])), 3),
               "minutes_total": round((time.monotonic() - started) / 60, 1)}  # fmt: skip
    with open(OUT / "translate_results.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (OUT / "translate_stats.json").write_text(json.dumps(summary, indent=2))
    print(f"🏁 {json.dumps(summary)}", flush=True)


if __name__ == "__main__":
    if DEVICE == "cuda":
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes"], check=False)
    main()
