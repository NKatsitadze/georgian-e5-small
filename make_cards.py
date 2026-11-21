"""Write the Hugging Face model cards (README.md) for both release folders from one template.

    uv run python make_cards.py
"""

from pathlib import Path

PROJECT = Path(__file__).resolve().parent
USER = "nkatsitadze"

VARIANTS = {
    "full": {
        "name": "georgian-e5-small",
        "languages": "- ka\n- en",
        "params": "118M parameters",
        "size": "458 MB",
        "score": "77.9",
        "row": "| **georgian-e5-small (this model)** | **118M** | **77.9** | 78.7 | 82.5 | 87.9 | 56.7 | 84.0 |\n"
        "| georgian-e5-small-lite | 58M | 77.9 | 78.7 | 82.5 | 87.9 | 56.6 | 84.0 |",
        "intro_extra": f"A [lite version](https://huggingface.co/{USER}/georgian-e5-small-lite) with a trimmed vocabulary "
        "(half the file size, same scores for Georgian, English and Russian) is also available.",
        "lite_section": "",
    },
    "lite": {
        "name": "georgian-e5-small-lite",
        "languages": "- ka\n- en\n- ru",
        "params": "118M parameters, 58M after this version's vocabulary trimming",
        "size": "223 MB",
        "score": "77.9",
        "row": f"| [georgian-e5-small](https://huggingface.co/{USER}/georgian-e5-small) | 118M | 77.9 | 78.7 | 82.5 | 87.9 | 56.7 | 84.0 |\n"
        "| **georgian-e5-small-lite (this model)** | **58M** | **77.9** | 78.7 | 82.5 | 87.9 | 56.6 | 84.0 |",
        "intro_extra": f"This is the **lite** version of [georgian-e5-small](https://huggingface.co/{USER}/georgian-e5-small): "
        "the same model with its vocabulary trimmed to Georgian, English and Russian, which halves the file size "
        "(458 MB → 223 MB) with the same benchmark scores.",
        "lite_section": """
## About the lite version

multilingual-e5-small's vocabulary has 250,002 word pieces for ~100 languages, and that vocabulary is 96M of the
model's 118M parameters. The lite version keeps 93,819 pieces: every piece used in a large sample of Georgian,
English and Russian text (Georgian Wikipedia and web text, English and Russian Wikipedia) plus every
single-character piece, so text in other scripts falls back to characters instead of becoming unknown.

The tokenizer picks the best split of a text from its vocabulary, so a text whose best split uses only kept
pieces is tokenized exactly as before and gets exactly the same vector. Checked on held-out text never used
to choose the vocabulary:

| held-out text | tokenized identically |
|---|---|
| Georgian (training-style texts) | 98.6% |
| Georgian web (FineWeb-2) | 98.1% |
| English Wikipedia | 97.5% |
| Russian Wikipedia | 98.5% |

For the remaining ~2%, the vectors stay very close to the full model's (mean cosine 0.995 for Georgian web
text, 0.97 for Russian), and no extra unknown tokens appear. **Use the full model for other languages.**
Speed is the same as the full model: the computation happens in the transformer layers, not the vocabulary.
""",
    },
}

TEMPLATE = """---
language:
{languages}
license: mit
library_name: sentence-transformers
pipeline_tag: sentence-similarity
base_model: intfloat/multilingual-e5-small
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- embeddings
- retrieval
- georgian
- knowledge-distillation
datasets:
- wikimedia/wikipedia
- HuggingFaceFW/fineweb-2
- PaDaS-Lab/webfaq
- tbilisi-ai-lab/wiki-ka-QA
---

# {name}

A small, fast text-embedding model for **Georgian (ქართული)** search and retrieval. It turns a text into a
384-number vector so that texts with similar meaning get similar vectors: the core of semantic search,
RAG (retrieval for LLM chatbots), FAQ matching and duplicate detection.

It is [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) ({params})
trained to imitate much larger models on Georgian. On the Georgian retrieval tasks of MTEB it scores
**{score} nDCG@10**, up from 61.5 for the original model, ahead of every other small model we tested and of
multilingual-e5-large, which is 5× larger. It runs comfortably on a CPU (~100 documents/second, ~6 ms per
query on a laptop), so documents can stay on your own server.

{intro_extra}

## Usage

```bash
pip install -U sentence-transformers
```

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("{user}/{name}")

queries = ["როდის დაარსდა თბილისი?"]
documents = [
    "თბილისი V საუკუნის მეორე ნახევარში დააარსა იბერიის მეფემ ვახტანგ გორგასალმა.",
    "ქუთაისი საქართველოს ერთ-ერთი უძველესი ქალაქია, რომელიც რიონის ნაპირზე მდებარეობს.",
]
query_vectors = model.encode(queries, prompt_name="query")         # adds "query: "
document_vectors = model.encode(documents, prompt_name="document")  # adds "passage: "
print(model.similarity(query_vectors, document_vectors))
# tensor([[0.7486, 0.5829]])  -> the first document answers the question
```

The model expects the e5 prefixes: **`"query: "`** before search queries and **`"passage: "`** before documents.
`prompt_name="query"` / `"document"` adds them for you; in other frameworks, add them to the text yourself.
Inputs longer than 512 tokens are truncated, so split long documents into passages.

## Results

nDCG@10 (0-100, higher is better) on the Georgian subsets of MTEB retrieval tasks. Belebele has three
directions: English question → Georgian passage, Georgian question → English passage, and Georgian →
Georgian.

| model | params | avg | Belebele en→ka | Belebele ka→en | Belebele ka→ka | GeorgianFAQ | WebFAQ ka |
|---|---|---|---|---|---|---|---|
| Snowflake/snowflake-arctic-embed-l-v2.0 | 568M | 85.5 | 85.9 | 89.2 | 91.0 | 69.5 | 91.8 |
| BAAI/bge-m3 (teacher) | 568M | 85.1 | 87.6 | 89.0 | 91.1 | 67.2 | 90.7 |
{row}
| l4sha/georgian-text-embeddings ¹ | 118M | 74.4 | 72.0 | 79.4 | 82.6 | 56.6 | 81.5 |
| intfloat/multilingual-e5-large | 560M | 73.4 | 77.5 | 87.5 | 81.8 | 45.1 | 74.9 |
| altaidevorg/bge-m3-distill-8l | | 70.3 | 64.9 | 76.1 | 76.6 | 53.6 | 80.3 |
| Vibels/georgian-e5-finetuned | | 69.8 | 67.3 | 78.6 | 79.1 | 47.5 | 76.2 |
| microsoft/harrier-oss-v1-270m | 270M | 68.9 | 64.5 | 81.2 | 73.5 | 49.1 | 76.4 |
| Qwen/Qwen3-Embedding-0.6B | 596M | 66.5 | 56.1 | 76.1 | 73.1 | 49.2 | 77.9 |
| intfloat/multilingual-e5-base | 278M | 64.1 | 61.3 | 78.9 | 75.6 | 38.7 | 66.2 |
| intfloat/multilingual-e5-small (starting point) ² | 118M | 61.5 | 57.2 | 72.8 | 67.0 | 41.4 | 69.1 |

¹ Trained on a Georgian translation of MS MARCO, whose license does not allow commercial use.
² Its best setting on Georgian (no prefixes); with the official prefixes it scores 57.3.

All models were evaluated by us with the same code (mteb 2.x), each with the prefix setting that worked best
for it. Speed on a laptop CPU (6 threads, Georgian FAQ documents): this model and multilingual-e5-small index
~102 documents/s; bge-m3, Snowflake-L and e5-large ~11 documents/s (9× slower).

## How it was trained

The student starts from multilingual-e5-small and learns from larger "teacher" models in two stages:

1. **Embedding distillation (6 epochs):** on 441,624 texts (Georgian Wikipedia, Georgian web text from
   FineWeb-2, Georgian WebFAQ, and some English Wikipedia), the student learns to reproduce the embeddings of
   [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3), compressed from 1024 to 384 dimensions with PCA.
2. **Ranking distillation (1 epoch):** on ~175k Georgian question → passage pairs (mostly
   [tbilisi-ai-lab/wiki-ka-QA](https://huggingface.co/datasets/tbilisi-ai-lab/wiki-ka-QA), plus WebFAQ), each with
   7 "hard negative" passages from other articles, the student learns to rank the passages the way a teacher
   does (KL divergence between score distributions), while continuing stage 1's embedding loss so it doesn't
   forget it. This was done twice, with two teachers: bge-m3 itself, and the cross-encoder
   [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3), which ranks Georgian passages more
   accurately (reranking bge-m3's results raised nDCG@10 from 85.4 to 89.8 in our test).
3. **Final model:** the average of the two stage-2 models' weights, which scored higher than either alone on
   all five tasks.

Training ran on free Kaggle GPUs (2× T4). Every training text that shared an 8-word sequence with any of the
11,941 benchmark test texts was removed before training.

## Limitations

- **The benchmark is small, and it also guided our choices.** The final recipe was picked among a handful of
  variants by its score on these tasks, so the numbers are slightly optimistic; expect the real gain over the
  best single run to be around half a point rather than exactly what the table shows.
- **Still behind the large models**, most of all on GeorgianFAQ (56.7 vs 67.2 for bge-m3). If you have a GPU and
  speed doesn't matter, bge-m3 or Snowflake-L is more accurate.
- **Trained and evaluated for Georgian and English.** Other languages keep whatever ability
  multilingual-e5-small had, but were not tested.
- **Training data:** wiki-ka-QA questions were generated by a language model (the dataset authors don't name
  it). A native speaker reviewed a sample of 50 before use and judged 82% good; the rest were mostly awkwardly
  phrased, not wrong. Web texts can contain the usual biases of web data.
{lite_section}
## License and credits

This model is released under the **MIT license**. It builds on:

| | license |
|---|---|
| [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) (base model) | MIT |
| [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) (teacher) | MIT |
| [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) (ranking teacher) | Apache-2.0 |
| [Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) (Georgian, English) | CC BY-SA 4.0 |
| [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) (kat_Geor) | ODC-By 1.0 |
| [WebFAQ](https://huggingface.co/datasets/PaDaS-Lab/webfaq) (Georgian) | CC BY 4.0 |
| [wiki-ka-QA](https://huggingface.co/datasets/tbilisi-ai-lab/wiki-ka-QA) by Tbilisi AI Lab | ODC-By |

Thanks to Tbilisi AI Lab for publishing wiki-ka-QA, and to the authors of e5, BGE, WebFAQ and FineWeb-2.
"""


def main() -> None:
    for folder, v in VARIANTS.items():
        text = TEMPLATE.format(user=USER, **v)
        (PROJECT / "data" / "release" / folder / "README.md").write_text(text, encoding="utf-8")
        print(f"✅ data/release/{folder}/README.md ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
