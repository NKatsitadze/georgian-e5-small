"""Kaggle GPU job: translate wiki-ka-QA questions into English with Qwen3-14B (AWQ) on vLLM.

Same prompt as the winning pilot (the passage is given as context; ~87% fully correct on 30
pilot questions). vLLM runs the model split over both T4s (tensor parallel), which is much faster
than transformers' generate.

LIMIT caps how many questions are translated (0 = all). Results are written every CHUNK questions,
and no new chunk starts after TIME_BUDGET_MIN, so a slow run still ends cleanly with its output.
Output (/kaggle/working): translations.parquet (id, query, en), translate_info.json.
Push from the project root:  uv run kaggle kernels push -p kaggle/translate_full
"""

import json
import subprocess
import sys
import time
from pathlib import Path

LIMIT = 20_000  # speed test (300 questions): 55/min on 2xT4, prompts ~1,000 tokens; 20k fits ~6 h
TIME_BUDGET_MIN = 420  # safety stop: no new chunk starts after this, so the run ends with its output saved
MODEL = "Qwen/Qwen3-14B-AWQ"  # Apache-2.0, official 4-bit AWQ weights
CHUNK = 1_000
MAX_PASSAGE_CHARS = 1_500
INPUT_ROOT = Path("/kaggle/input")
OUT = Path("/kaggle/working")

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


def clean(raw: str) -> str:
    text = raw.split("</think>")[-1].strip()
    line = next((part.strip() for part in text.splitlines() if part.strip()), "")
    return line.strip(" \"'„“«»*").strip()


def main() -> None:
    started = time.monotonic()
    import cutlass  # noqa: F401  # fail early and clearly if the warm-up dependency is missing
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    import vllm
    from vllm import LLM, SamplingParams

    gpus = torch.cuda.device_count()
    print(f"{gpus} GPU(s) {torch.cuda.get_device_name(0) if gpus else ''} | torch {torch.__version__} | vllm {vllm.__version__}"
          f" | install {(time.monotonic() - started) / 60:.1f} min", flush=True)  # fmt: skip
    rows = pq.read_table(sorted(INPUT_ROOT.rglob("to_translate.parquet"))[0]).to_pylist()
    if LIMIT:
        rows = rows[:LIMIT]

    llm = LLM(model=MODEL, tensor_parallel_size=max(1, gpus), dtype="half", max_model_len=2048,
              gpu_memory_utilization=0.90, seed=0)  # fmt: skip
    params = SamplingParams(temperature=0.0, max_tokens=96)
    print(f"✅ model loaded at {(time.monotonic() - started) / 60:.1f} min", flush=True)

    done: list[dict] = []
    gen_started = time.monotonic()
    for start in range(0, len(rows), CHUNK):
        if (time.monotonic() - started) / 60 > TIME_BUDGET_MIN:
            print(f"⏱️ time budget reached: stopping after {len(done)} questions", flush=True)
            break
        chunk = rows[start : start + CHUNK]
        conversations = [[{"role": "user", "content": PROMPT.format(passage=r["passage"][:MAX_PASSAGE_CHARS], question=r["query"])}]
                         for r in chunk]  # fmt: skip
        outputs = llm.chat(conversations, params, chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)
        done += [{"id": r["id"], "query": r["query"], "en": clean(o.outputs[0].text)} for r, o in zip(chunk, outputs, strict=True)]
        pq.write_table(pa.Table.from_pylist(done), OUT / "translations.parquet")  # rewritten each chunk: partial runs stay usable
        rate = len(done) / (time.monotonic() - gen_started) * 60
        print(f"   {len(done)}/{len(rows)} | {rate:.0f} questions/min | ~{(len(rows) - len(done)) / rate:.0f} min left", flush=True)

    info = {"model": MODEL, "questions": len(done), "empty": sum(not d["en"] for d in done),
            "questions_per_minute": round(len(done) / (time.monotonic() - gen_started) * 60, 1),
            "gpus": gpus, "minutes_total": round((time.monotonic() - started) / 60, 1)}  # fmt: skip
    (OUT / "translate_info.json").write_text(json.dumps(info, indent=2))
    print(f"🏁 {json.dumps(info)}", flush=True)
    for d in done[:5]:
        print(f"   {d['query']}  ->  {d['en']}", flush=True)


if __name__ == "__main__":
    if "--work" in sys.argv:
        main()
    else:
        # Install first, then run in a FRESH interpreter: packages installed into a running Python
        # (and the worker processes vLLM forks from it) are not always importable, e.g. `cutlass`.
        # nvidia-cutlass-dsl provides `cutlass`, which vLLM 0.34's kernel warm-up imports but doesn't install.
        install = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm", "nvidia-cutlass-dsl"], capture_output=True, text=True)
        print(f"pip install: exit {install.returncode}", flush=True)
        sys.exit(subprocess.run([sys.executable, __file__, "--work"]).returncode)
