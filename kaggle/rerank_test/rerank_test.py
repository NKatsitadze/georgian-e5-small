"""Kaggle GPU job: is a reranker a better teacher than bge-m3 for Georgian?

1. bge-m3 retrieves the top TOP_K passages for every test question (nDCG@10 of bge-m3 alone is also
   computed on ALL questions, as a sanity check against our MTEB numbers, e.g. 85.1 average).
2. Each reranker re-orders bge-m3's top candidates for a fixed random sample of questions
   (up to SAMPLE per test), and we compare nDCG@10 with bge-m3 alone on the same sample.
   Both GPUs work at once: GPU 0 runs the cross-encoders, GPU 1 runs Qwen3-Reranker-4B.

All candidate rerankers are Apache-2.0. Output (/kaggle/working): rerank_results.json.
Push from the project root:  uv run kaggle kernels push -p kaggle/rerank_test
"""

import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

RETRIEVER = "BAAI/bge-m3"
TOP_K = 20  # candidates per question reranked by the cross-encoders
QWEN_TOP_K = 10  # Qwen3-Reranker-4B is slow on Georgian (~1 token per character), so its top 10
SAMPLE = 300  # questions per test subset for reranking (WebFAQ has 241)
SEED = 42
INPUT_ROOT = Path(os.environ.get("RR_INPUT", "/kaggle/input"))
OUT = Path(os.environ.get("RR_OUTPUT", "/kaggle/working"))
LIMIT = int(os.environ.get("RR_LIMIT", "0"))  # smoke test: tiny sample
CROSS_ENCODERS = os.environ.get("RR_CROSS", "BAAI/bge-reranker-v2-m3,Alibaba-NLP/gte-multilingual-reranker-base").split(",")
QWEN_RERANKER = os.environ.get("RR_QWEN", "Qwen/Qwen3-Reranker-4B")
QWEN_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"


def ndcg_at_10(ranked: list[str], qrels: dict[str, int]) -> float:
    dcg = sum(qrels.get(doc, 0) / math.log2(i + 2) for i, doc in enumerate(ranked[:10]))
    ideal = sorted(qrels.values(), reverse=True)[:10]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def retrieve() -> None:
    from sentence_transformers import SentenceTransformer

    subsets = json.loads(sorted(INPUT_ROOT.rglob("test_tasks.json"))[0].read_text(encoding="utf-8"))
    model = SentenceTransformer(RETRIEVER, device="cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        model.half()
    model.max_seq_length = 512
    rng = random.Random(SEED)
    out, full_scores = [], {}
    for s in subsets:
        qids, dids = list(s["queries"]), list(s["corpus"])
        if LIMIT:
            qids = qids[:LIMIT]
        q = model.encode([s["queries"][i] for i in qids], batch_size=64, normalize_embeddings=True, convert_to_tensor=True)
        d = model.encode([s["corpus"][i] for i in dids], batch_size=64, normalize_embeddings=True, convert_to_tensor=True)
        scores = (q @ d.T).float()
        top = scores.topk(min(TOP_K, len(dids)), dim=1).indices.cpu().tolist()
        ranked = {qid: [dids[j] for j in row] for qid, row in zip(qids, top, strict=True)}
        name = f"{s['task']}/{s['subset']}"
        full_scores[name] = round(100 * float(np.mean([ndcg_at_10(ranked[qid], s["qrels"].get(qid, {})) for qid in qids])), 2)
        sample = rng.sample(qids, min(SAMPLE, len(qids)))
        out.append({"name": name, "items": [{"qid": qid, "query": s["queries"][qid], "qrels": s["qrels"].get(qid, {}),
                                              "candidates": [[did, s["corpus"][did]] for did in ranked[qid]]} for qid in sample]})  # fmt: skip
        print(f"✅ {name}: bge-m3 nDCG@10 {full_scores[name]} on all {len(qids)} questions", flush=True)
    (OUT / "candidates.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    (OUT / "bge_m3_full.json").write_text(json.dumps(full_scores, indent=2))


def evaluate(sets: list[dict], score_fn, top_k: int, label: str) -> dict:
    """Rerank each question's top_k candidates with score_fn(query, docs) -> list[float]."""
    result, started = {}, time.monotonic()
    for s in sets:
        base, reranked = [], []
        for item in s["items"]:
            cands = item["candidates"][:top_k]
            scores = score_fn(item["query"], [text for _, text in cands])
            order = [cands[i][0] for i in np.argsort(-np.asarray(scores), kind="stable")]
            reranked.append(ndcg_at_10(order, item["qrels"]))
            base.append(ndcg_at_10([d for d, _ in item["candidates"]], item["qrels"]))
        result[s["name"]] = {"bge_m3": round(100 * float(np.mean(base)), 2), "reranked": round(100 * float(np.mean(reranked)), 2),
                             "questions": len(s["items"])}  # fmt: skip
        print(f"   {label} | {s['name']}: bge-m3 {result[s['name']]['bge_m3']} -> {result[s['name']]['reranked']}"
              f" | {(time.monotonic() - started) / 60:.1f} min", flush=True)  # fmt: skip
    result["average"] = {k: round(float(np.mean([v[k] for n, v in result.items() if n != "average"])), 2) for k in ("bge_m3", "reranked")}
    return result


def run_cross_encoders() -> None:
    from sentence_transformers import CrossEncoder

    sets = json.loads((OUT / "candidates.json").read_text(encoding="utf-8"))
    results = {}
    for repo in CROSS_ENCODERS:
        print(f"▶ {repo}", flush=True)
        try:
            model = CrossEncoder(repo, max_length=512, device="cuda" if torch.cuda.is_available() else "cpu",
                                 trust_remote_code=True, model_kwargs={"torch_dtype": torch.float16} if torch.cuda.is_available() else {})  # fmt: skip
            score = lambda q, docs: model.predict([(q, d) for d in docs], batch_size=32, show_progress_bar=False)  # noqa: E731
            results[repo] = {f"top{k}": evaluate(sets, score, k, f"{repo} top{k}") for k in (QWEN_TOP_K, TOP_K)}
        except Exception as error:  # one failing model must not lose the others' results
            print(f"❌ {repo}: {type(error).__name__}: {error}", flush=True)
            results[repo] = {"error": f"{type(error).__name__}: {error}"}
        (OUT / "cross_results.json").write_text(json.dumps(results, indent=2))


def run_qwen() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sets = json.loads((OUT / "candidates.json").read_text(encoding="utf-8"))
    print(f"▶ {QWEN_RERANKER}", flush=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(QWEN_RERANKER, padding_side="left")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(QWEN_RERANKER, dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()
        yes, no = tokenizer.convert_tokens_to_ids("yes"), tokenizer.convert_tokens_to_ids("no")
        prefix = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. "
                  "Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")  # fmt: skip
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

        @torch.no_grad()
        def score(query: str, docs: list[str]) -> list[float]:
            texts = [f"{prefix}<Instruct>: {QWEN_INSTRUCTION}\n<Query>: {query}\n<Document>: {d[:1500]}{suffix}" for d in docs]
            out = []
            for start in range(0, len(texts), 5):
                batch = tokenizer(texts[start : start + 5], padding=True, truncation=True, max_length=1600, return_tensors="pt").to(device)
                logits = model(**batch).logits[:, -1, :]
                pair = torch.stack([logits[:, no], logits[:, yes]], dim=1).float()
                out += torch.log_softmax(pair, dim=1)[:, 1].exp().tolist()
            return out

        results = {QWEN_RERANKER: {f"top{QWEN_TOP_K}": evaluate(sets, score, QWEN_TOP_K, f"qwen top{QWEN_TOP_K}")}}
    except Exception as error:
        print(f"❌ {QWEN_RERANKER}: {type(error).__name__}: {error}", flush=True)
        results = {QWEN_RERANKER: {"error": f"{type(error).__name__}: {error}"}}
    (OUT / "qwen_results.json").write_text(json.dumps(results, indent=2))


def main() -> None:
    started = time.monotonic()
    retrieve()
    gpus = torch.cuda.device_count()
    procs = []
    for role, gpu in (("cross", 0), ("qwen", 1 if gpus > 1 else 0)):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)} if gpus else dict(os.environ)
        procs.append(subprocess.Popen([sys.executable, __file__, f"--{role}"], env=env))
        if gpus < 2:
            procs[-1].wait()  # one GPU (or CPU): run them one after the other
    for p in procs:
        p.wait()
    merged = {"bge_m3_all_questions": json.loads((OUT / "bge_m3_full.json").read_text())}
    for name in ("cross_results.json", "qwen_results.json"):
        if (OUT / name).exists():
            merged.update(json.loads((OUT / name).read_text()))
    merged["minutes_total"] = round((time.monotonic() - started) / 60, 1)
    (OUT / "rerank_results.json").write_text(json.dumps(merged, indent=2))
    print(f"🏁 {json.dumps(merged)}", flush=True)


if __name__ == "__main__":
    if "--cross" in sys.argv:
        run_cross_encoders()
    elif "--qwen" in sys.argv:
        run_qwen()
    else:
        main()
