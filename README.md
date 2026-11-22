# georgian-e5-small

A small, fast **Georgian text-embedding model** for semantic search and RAG, and the full pipeline that
built it: knowledge distillation from large models, run on free Kaggle GPUs.

- 🤗 Model: [nkatsitadze/georgian-e5-small](https://huggingface.co/nkatsitadze/georgian-e5-small) (118M params)
- 🤗 Lite: [nkatsitadze/georgian-e5-small-lite](https://huggingface.co/nkatsitadze/georgian-e5-small-lite) (same scores, half the file size)
- 🔎 Demo: [Georgian semantic search](https://huggingface.co/spaces/nkatsitadze/georgian-e5-small-demo), the lite model running **in your browser** (Transformers.js, 8-bit ONNX)

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("nkatsitadze/georgian-e5-small")
q = model.encode(["როდის დაარსდა თბილისი?"], prompt_name="query")
d = model.encode(["თბილისი V საუკუნის მეორე ნახევარში დააარსა ვახტანგ გორგასალმა."], prompt_name="document")
print(model.similarity(q, d))
```

## Results

nDCG@10 on the Georgian retrieval tasks of MTEB (Belebele ×3 directions, GeorgianFAQ, WebFAQ), all models
evaluated with the same code:

| model | params | avg nDCG@10 | CPU docs/s |
|---|---|---|---|
| BAAI/bge-m3 (teacher) | 568M | 85.1 | 11 |
| **georgian-e5-small** | **118M** | **77.9** | **102** |
| l4sha/georgian-text-embeddings | 118M | 74.4 | 102 |
| intfloat/multilingual-e5-large | 560M | 73.4 | 11 |
| intfloat/multilingual-e5-small (starting point) | 118M | 61.5 | 102 |

+16.4 points over the model it started from, ahead of every small model tested and of e5-large (5× larger),
at 9× the CPU speed of the teacher. Full table with 13 models: [results/baseline.md](results/baseline.md).

## How it was built

1. **Benchmark first** ([baseline.py](baseline.py), [speed.py](speed.py)): measured every Georgian-capable
   embedding model on quality and CPU speed, and set the goal: big-model quality at small-model speed.
2. **Data** ([build_dataset.py](build_dataset.py)): 441k Georgian/English texts from Wikipedia, FineWeb-2 and
   WebFAQ, deduplicated, with every text sharing an 8-word sequence with the test sets removed.
3. **Embedding distillation** ([kaggle/distill](kaggle/distill)): the student learns to reproduce bge-m3's
   embeddings (PCA 1024 → 384). Multi-GPU DDP, token-length batching (padding 38% → 0.6%), fp16.
4. **Ranking distillation** ([kaggle/rank_v3](kaggle/rank_v3), [kaggle/teacher_pairs](kaggle/teacher_pairs)):
   on ~175k Georgian question → passage pairs with teacher-mined hard negatives, the student matches the
   teacher's score distribution (KL), while keeping the embedding loss. Procrustes alignment of the PCA targets,
   exact micro-batch gradient accumulation to fit T4 memory.
5. **Reranker teacher** ([kaggle/rerank_test](kaggle/rerank_test), [kaggle/teacher_pairs_v6](kaggle/teacher_pairs_v6),
   [kaggle/rank_v6](kaggle/rank_v6)): bge-reranker-v2-m3 ranks Georgian passages better than bge-m3
   (85.4 → 89.8 when reranking), so it re-scored the training pairs, with its temperature calibrated to match
   the original targets' sharpness.
6. **Model soup**: the final model averages the weights of the bge-m3-taught and reranker-taught runs.
7. **Release** ([trim_vocab.py](trim_vocab.py), [make_cards.py](make_cards.py)): vocabulary trimming for the lite
   version, verified by identical tokenization on held-out text; configs in the classic format so
   sentence-transformers 3.x–6.x can load it; ONNX weights (fp32 identical to PyTorch, 8-bit 4× smaller at about
   −1 nDCG point), checked end to end through Transformers.js.
8. **Demo** ([demo/](demo)): a static page that runs the 8-bit lite model in the browser over 30,000 Georgian
   Wikipedia passages (int8 document vectors, gzipped passages, ~25 ms per search). A Gradio version is in
   `demo/space/`.

## Experiment log

Each run changed one thing, and every result was benchmarked, including the ones that didn't help:

| run | change | avg | verdict |
|---|---|---|---|
| e5-small | starting point | 61.5 | |
| v1 | embedding distillation, 2 epochs | 60.6 | |
| v2 | same, 6 epochs | 72.6 | curve flattened at epoch 6 |
| v3 | + ranking distillation on question → passage pairs | 77.4 | ✅ the big win |
| v4 | harder negatives (looser false-negative guard, 7 negatives) | 77.1 | ✗ no gain |
| v5 | + 18k English questions translated by Qwen3-14B | 77.2 | ✗ held-out English 74 → 84%, benchmark unchanged |
| v6 | reranker (bge-reranker-v2-m3) as ranking teacher | 77.7 | small gain |
| soup | weight average of v3 and v6 | 78.0 | ✅ better than both on all 5 tasks |
| release | soup with 512-token inputs | 77.9 | published |

Also tried and dropped: LLM-generated Georgian questions (a native speaker rated only 18% acceptable, while an
automatic round-trip check passed 92%), longer input limits, averaging epochs of one run.

## Data and licensing

Everything used allows others to use the model, commercially included: base model and teachers are
MIT / Apache-2.0; data is Wikipedia (CC BY-SA 4.0), FineWeb-2 (ODC-By), WebFAQ (CC BY 4.0) and
[wiki-ka-QA](https://huggingface.co/datasets/tbilisi-ai-lab/wiki-ka-QA) by Tbilisi AI Lab (ODC-By). Data generated
with models whose terms restrict training on their outputs was avoided, and datasets with non-commercial
licenses (e.g. MS MARCO) were not used.

## Repository layout

| path | what |
|---|---|
| `baseline.py`, `speed.py`, `pca_ceiling.py` | benchmarks: MTEB Georgian retrieval, CPU speed, PCA ceiling |
| `build_dataset.py` | builds the 441k-text distillation set with test-overlap removal |
| `qgen/` | question → passage pairs (wiki-ka-QA, WebFAQ), translation pilots, test-set export |
| `kaggle/` | GPU jobs, one folder per Kaggle kernel (teacher embeddings, distillation, ranking runs, reranker scoring) |
| `trim_vocab.py`, `make_cards.py` | release: lite version and model cards |
| `demo/` | the browser demo (`static/`, published as the Hugging Face Space), a Gradio version (`space/`) and the scripts that build the search collection |
| `review/` | the page used for native-speaker review of generated training data |
| `results/` | benchmark tables and logs |

## Reproducing

Requires [uv](https://docs.astral.sh/uv/) and a Kaggle account (free 2× T4 GPUs).

```bash
uv sync
uv run python baseline.py --models intfloat/multilingual-e5-small   # benchmark any model
uv run python build_dataset.py                                       # build training texts
# then upload data/kaggle/* as private Kaggle datasets and push the jobs in kaggle/ in order:
# teacher → distill → teacher_pairs → rank_v3, and teacher_pairs_v6 → rank_v6
uv run kaggle kernels push -p kaggle/teacher
```

Each `kaggle/*/` script has a local smoke-test mode (environment variables at the top of the file).

## License

Code: MIT. Model: MIT (see the model card for data attributions).
