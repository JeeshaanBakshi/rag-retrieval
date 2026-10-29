# RAG Retrieval

Search a document by meaning, not keywords — the retrieval half of RAG, built from scratch in Python, with a website and live demo.

```
read text → overlapping chunks → all-MiniLM-L6-v2 embeddings → cosine similarity → top 3
```

**Case study:** [jeeshaanbakshi.vercel.app/projects/rag-semantic-search](https://jeeshaanbakshi.vercel.app/projects/rag-semantic-search). The search bar on that site runs this same pipeline (same model) in the browser.

## Run it

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open **http://localhost:3000**. The first run downloads the model (~90 MB).

## Files

| File | What it is |
|---|---|
| `app.py` | The pipeline (steps 1–5) and a small web server |
| `index.html` | The website: search box, top 3 results, and the full document with the matches highlighted |
| `text.txt` | The document being searched (an article on NeRF) |

## Built with

- **Retrieval pipeline** (`app.py`, steps 1–5): written by me in Python.
- **Web front end** (`index.html`): built with help from [Claude Code](https://claude.com/claude-code), an AI coding assistant.

## How it works

1. **Read** `text.txt` into `content`.
2. **Chunk** it into `cs`: windows of `size` characters that share `overlap` characters, so an idea cut at one edge is whole in the next chunk.
3. **Load** `all-MiniLM-L6-v2` (SentenceTransformers).
4. **Embed** every chunk into `chunk_emb` — 384 numbers each.
5. **Score** the question's embedding `qem` against every chunk with cosine similarity, sort `topscindex` highest first, keep the top 3 as `highl`.

## Example

Question: *How does NeRF represent a 3D scene?* (size 100, overlap 40 → 61 chunks)

| score | chunk |
|---:|---|
| 0.690 | "the core innovations in NeRF is the use of volumetric rendering, where a ray is cast from the camera" |
| 0.670 | "…the adaptation of NeRF-like models for dynamic scenes, where the geometry and appea…" |
| 0.640 | "…novel views of complex 3D scenes. Unlike traditional graphics pipelines that rely on explicit" |

## Limits and next steps

- Fixed-width chunks cut words in half — sentence-aware chunking would fix that.
- Retrieval only: next is passing the top passages to an LLM to write a grounded answer.
