"""Georgian semantic search demo for nkatsitadze/georgian-e5-small (Hugging Face Space, Gradio)."""

import html

import gradio as gr
import numpy as np
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer

OURS, BASE = "nkatsitadze/georgian-e5-small", "intfloat/multilingual-e5-small"
TOP_K = 5

corpus = pq.read_table("corpus.parquet").to_pylist()
vectors = {"ours": np.load("emb_ours.npy").astype(np.float32), "base": np.load("emb_base.npy").astype(np.float32)}
models = {"ours": SentenceTransformer(OURS, device="cpu"), "base": SentenceTransformer(BASE, device="cpu")}


def embed(key: str, texts: list[str], kind: str) -> np.ndarray:
    """Our model uses its e5 prefixes; the original model gets its best Georgian setting (no prefixes)."""
    if key == "ours":
        return models[key].encode(texts, prompt_name=kind, normalize_embeddings=True)
    return models[key].encode(texts, normalize_embeddings=True)


def render(indices: np.ndarray, scores: np.ndarray) -> str:
    cards = []
    for rank, i in enumerate(indices, start=1):
        doc = corpus[i]
        snippet = doc["text"] if len(doc["text"]) <= 320 else doc["text"][:320].rsplit(" ", 1)[0] + " …"
        cards.append(
            f'<div class="hit"><div class="hit-top"><span class="rank">{rank}</span>'
            f'<a href="{html.escape(doc["url"])}" target="_blank" rel="noopener">{html.escape(doc["title"])}</a>'
            f'<span class="score">{scores[i]:.3f}</span></div><p lang="ka">{html.escape(snippet)}</p></div>'
        )
    return "".join(cards)


def search(query: str) -> tuple[str, str]:
    query = (query or "").strip()
    if not query:
        return "", ""
    results = []
    for key in ("ours", "base"):
        scores = vectors[key] @ embed(key, [query], "query")[0]
        results.append(render(np.argsort(-scores)[:TOP_K], scores))
    return results[0], results[1]


def compare(text_a: str, text_b: str) -> str:
    if not (text_a or "").strip() or not (text_b or "").strip():
        return ""
    # Only our model: the original model scores almost every pair high (even unrelated ones), so its numbers
    # would look "better" side by side while meaning less.
    a, b = embed("ours", [text_a, text_b], "query")
    similarity = float(a @ b)
    verdict = "very similar" if similarity >= 0.8 else "related" if similarity >= 0.7 else "probably unrelated"
    return (f'<div class="sim"><span>georgian-e5-small</span><div class="bar"><div style="width:{max(0.0, similarity) * 100:.0f}%">'
            f"</div></div><b>{similarity:.3f}</b></div><p class=\"note\">{verdict} · as a guide: related texts usually score "
            f"0.8–0.9, unrelated ones around 0.55.</p>")  # fmt: skip


CSS = """
.hit { border: 1px solid var(--border-color-primary); border-radius: 10px; padding: 10px 14px; margin-bottom: 10px;
       background: var(--block-background-fill); }
.hit-top { display: flex; align-items: center; gap: 10px; }
.hit-top a { font-weight: 600; text-decoration: none; flex: 1; overflow-wrap: anywhere; }
.rank { font-size: 12px; opacity: .6; min-width: 14px; }
.score { font-variant-numeric: tabular-nums; font-size: 13px; opacity: .75; }
.hit p { margin: 6px 0 0; line-height: 1.6; font-size: 14.5px; overflow-wrap: anywhere; }
.sim { display: grid; grid-template-columns: minmax(120px, 1.2fr) 2fr 60px; gap: 12px; align-items: center; margin: 10px 0; }
.bar { height: 10px; border-radius: 6px; background: var(--border-color-primary); overflow: hidden; }
.bar div { height: 100%; background: var(--color-accent); }
.sim b { font-variant-numeric: tabular-nums; text-align: right; }
.note { opacity: .75; font-size: 14px; margin: 4px 0 0; }
"""

INTRO = """
# 🇬🇪 Georgian semantic search · georgian-e5-small

Type a question in **Georgian** (or English) and see which passages the model finds among the opening
paragraphs of **30,000 Georgian Wikipedia articles**. Results are matched by meaning, not by exact words.
Left: [**georgian-e5-small**](https://huggingface.co/nkatsitadze/georgian-e5-small), our model.
Right: the model it started from, [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small).

დასვით კითხვა ქართულად (ან ინგლისურად) — მოდელი ტექსტებს მნიშვნელობით პოულობს და არა მხოლოდ სიტყვების დამთხვევით.
"""

FOOTER = """
---
**georgian-e5-small** · 118M parameters · runs on a CPU · MIT license ·
Georgian MTEB retrieval: **77.9** nDCG@10 (original model 61.5, bge-m3 85.1) ·
[model card](https://huggingface.co/nkatsitadze/georgian-e5-small) ·
[lite version](https://huggingface.co/nkatsitadze/georgian-e5-small-lite)

Passages from [Georgian Wikipedia](https://ka.wikipedia.org) (CC BY-SA 4.0); each result links to its article.
Scores are cosine similarities: compare them within one model's list, not between models.
"""

EXAMPLES = [
    "როდის დაარსდა თბილისი?",
    "ვინ დაწერა ვეფხისტყაოსანი?",
    "ვინ იყო ილია ჭავჭავაძე?",
    "რა არის შავი ხვრელი?",
    "როდის მოხდა დიდგორის ბრძოლა?",
    "Who was Queen Tamar of Georgia?",
    "Where is the Gelati monastery?",
]

with gr.Blocks(title="Georgian semantic search") as demo:
    gr.Markdown(INTRO)
    with gr.Tab("Search · ძიება"):
        with gr.Row():
            query = gr.Textbox(label="Question · კითხვა", placeholder="მაგ.: როდის დაარსდა თბილისი?", scale=5, autofocus=True)
            button = gr.Button("Search", variant="primary", scale=1)
        gr.Examples(EXAMPLES, inputs=query)
        with gr.Row():
            with gr.Column():
                gr.Markdown("### georgian-e5-small (ours)")
                ours_out = gr.HTML()
            with gr.Column():
                gr.Markdown("### multilingual-e5-small (original)")
                base_out = gr.HTML()
        button.click(search, query, [ours_out, base_out])
        query.submit(search, query, [ours_out, base_out])
    with gr.Tab("Compare two texts · ორი ტექსტის შედარება"):
        with gr.Row():
            text_a = gr.Textbox(label="Text A", value="ქალაქი, სადაც საქართველოს პარლამენტი მდებარეობს")
            text_b = gr.Textbox(label="Text B", value="თბილისი საქართველოს დედაქალაქია")
        compare_button = gr.Button("Compare", variant="primary")
        compare_out = gr.HTML()
        compare_button.click(compare, [text_a, text_b], compare_out)
    gr.Markdown(FOOTER)

if __name__ == "__main__":
    demo.launch(css=CSS, theme=gr.themes.Soft())
