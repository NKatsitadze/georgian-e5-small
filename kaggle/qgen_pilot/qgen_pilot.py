"""Kaggle GPU job: which open (Apache-2.0) LLM writes good Georgian search questions?

Each candidate model writes one Georgian search question for each of the 50 pilot passages. Then
the teacher (bge-m3) does a round-trip check: it must find the question's source passage among
5,050 passages. A human (native speaker) reviews the same pairs blind, so the automatic check can be
calibrated against human judgment before any large-scale generation.

Output (/kaggle/working): pilot_results.jsonl (one row per passage x model) and pilot_stats.json.
Push from the project root:  uv run kaggle kernels push -p kaggle/qgen_pilot
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

# Each candidate: (name, Hugging Face repo, load in 4-bit). Qwen3 is Apache-2.0, so generated text
# may be used to train and publish a model without extra conditions.
CANDIDATES = [
    ("qwen3-8b", "Qwen/Qwen3-8B", False),  # fp16 (~16 GB), split over the two T4s
    ("qwen3-14b-4bit", "Qwen/Qwen3-14B", True),  # 4-bit NF4 (~9 GB)
]
TEACHER = "BAAI/bge-m3"
PROMPT = """Below is a passage in Georgian. Write ONE question in natural Georgian that a real person might type into a search engine, and that this passage answers.

Rules:
- Write only in Georgian (Mkhedruli script). Names and terms may stay as they appear in the passage.
- 4 to 15 words. A real question, not a title.
- Use your own words; do not copy long phrases from the passage.
- Be specific enough that this passage is a good answer, but never mention "the passage", "the text" or "the article".
- Return only the question, nothing else.

Passage:
{passage}"""
GENERATION = {"max_new_tokens": 96, "do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20}  # Qwen3 non-thinking defaults
BATCH_SIZE = 8
SEED = 42

# Kaggle defaults; the local smoke test overrides them (tiny model, few passages, CPU).
INPUT_ROOT = Path(os.environ.get("QGEN_INPUT", "/kaggle/input"))
OUT = Path(os.environ.get("QGEN_OUTPUT", "/kaggle/working"))
LIMIT = int(os.environ.get("QGEN_LIMIT", "0"))
if os.environ.get("QGEN_CANDIDATES"):  # e.g. "tiny=Qwen/Qwen3-0.6B"
    CANDIDATES = [(name, repo, False) for name, repo in (c.split("=") for c in os.environ["QGEN_CANDIDATES"].split(","))]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def find_input(name: str) -> Path:
    matches = sorted(INPUT_ROOT.rglob(name))
    if not matches:
        sys.exit(f"❌ {name} not found under {INPUT_ROOT}")
    return matches[0]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def clean_question(raw: str) -> str:
    """First non-empty line, without quotes, numbering or a leftover thinking block."""
    text = raw.split("</think>")[-1].strip()
    line = next((part.strip() for part in text.splitlines() if part.strip()), "")
    return line.strip(" \"'„“«»*-–1234567890.").strip()


def generate_questions(repo: str, four_bit: bool, passages: list[dict]) -> tuple[list[str], dict]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(repo, padding_side="left")
    options: dict = {"dtype": torch.float16} if DEVICE == "cuda" else {"dtype": torch.float32}
    if DEVICE == "cuda":
        options["device_map"] = "auto"  # spreads the model over every GPU it is given
    if four_bit:
        from transformers import BitsAndBytesConfig

        options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
        )
    model = AutoModelForCausalLM.from_pretrained(repo, **options)
    model.eval()

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPT.format(passage=p["text"])}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        for p in passages
    ]  # fmt: skip
    questions, input_tokens, output_tokens = [], 0, 0
    started = time.monotonic()
    torch.manual_seed(SEED)
    for start in range(0, len(prompts), BATCH_SIZE):
        batch = tokenizer(prompts[start : start + BATCH_SIZE], return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            output = model.generate(**batch, **GENERATION, pad_token_id=tokenizer.pad_token_id)
        new_tokens = output[:, batch["input_ids"].shape[1] :]
        input_tokens += int(batch["attention_mask"].sum())
        output_tokens += int((new_tokens != tokenizer.pad_token_id).sum())
        questions += [clean_question(t) for t in tokenizer.batch_decode(new_tokens, skip_special_tokens=True)]
        print(f"   {min(start + BATCH_SIZE, len(prompts))}/{len(prompts)} passages", flush=True)
    seconds = time.monotonic() - started
    stats = {
        "seconds": round(seconds, 1),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "passages_per_minute": round(len(passages) / seconds * 60, 1),
    }
    del model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return questions, stats


def round_trip_ranks(rows: list[dict], passages: list[dict], distractors: list[dict]) -> list[int]:
    """Rank (1 = best) of each question's own passage among pilot passages + distractors."""
    from sentence_transformers import SentenceTransformer

    teacher = SentenceTransformer(TEACHER, device=DEVICE)
    if DEVICE == "cuda":
        teacher.half()
    pool = [p["text"] for p in passages] + [d["text"] for d in distractors]
    pool_vectors = teacher.encode(pool, batch_size=64, normalize_embeddings=True, convert_to_tensor=True)
    question_vectors = teacher.encode([r["question"] or "-" for r in rows], batch_size=64,
                                      normalize_embeddings=True, convert_to_tensor=True)  # fmt: skip
    scores = (question_vectors @ pool_vectors.T).float().cpu().numpy()
    index_of = {p["id"]: i for i, p in enumerate(passages)}
    return [int((scores[i] > scores[i, index_of[r["id"]]]).sum()) + 1 for i, r in enumerate(rows)]


def main() -> None:
    started = time.monotonic()
    print(f"device {DEVICE} x {torch.cuda.device_count() if DEVICE == 'cuda' else 1} | torch {torch.__version__}", flush=True)
    passages = read_jsonl(find_input("pilot_passages.jsonl"))
    distractors = read_jsonl(find_input("distractors.jsonl"))
    if LIMIT:
        passages, distractors = passages[:LIMIT], distractors[: LIMIT * 20]

    rows, model_stats = [], {}
    for name, repo, four_bit in CANDIDATES:
        print(f"▶ {name} ({repo}{', 4-bit' if four_bit else ''})", flush=True)
        try:
            questions, stats = generate_questions(repo, four_bit, passages)
        except Exception as error:  # one failing candidate must not lose the other's results
            print(f"❌ {name}: {type(error).__name__}: {error}", flush=True)
            model_stats[name] = {"error": f"{type(error).__name__}: {error}"}
            continue
        model_stats[name] = stats
        rows += [{"id": p["id"], "source": p["source"], "passage": p["text"], "model": name, "question": q}
                 for p, q in zip(passages, questions, strict=True)]  # fmt: skip
        print(f"✅ {name}: {stats}", flush=True)

    if not rows:
        sys.exit("❌ no candidate produced questions")
    print(f"▶ round-trip check with {TEACHER} over {len(passages) + len(distractors)} passages", flush=True)
    for row, rank in zip(rows, round_trip_ranks(rows, passages, distractors), strict=True):
        row["teacher_rank"] = rank
    for name in model_stats:
        ranks = [r["teacher_rank"] for r in rows if r["model"] == name]
        if ranks:
            model_stats[name].update({
                "round_trip_top1": round(float(np.mean([r == 1 for r in ranks])), 3),
                "round_trip_top10": round(float(np.mean([r <= 10 for r in ranks])), 3),
                "empty_questions": sum(not r["question"] for r in rows if r["model"] == name),
            })  # fmt: skip

    with open(OUT / "pilot_results.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"models": model_stats, "passages": len(passages), "pool": len(passages) + len(distractors),
               "minutes_total": round((time.monotonic() - started) / 60, 1)}  # fmt: skip
    (OUT / "pilot_stats.json").write_text(json.dumps(summary, indent=2))
    print(f"🏁 {json.dumps(summary)}", flush=True)


if __name__ == "__main__":
    if DEVICE == "cuda":
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes"], check=False)
    main()
