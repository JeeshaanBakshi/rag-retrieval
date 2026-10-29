# RAG Retrieval

Find the passages in a document that answer a question — by meaning, not keywords.
This is the retrieval half of RAG (retrieval-augmented generation).

```
document → overlapping chunks → 384-d embeddings → cosine similarity → top-k passages
```

**Live demo:** the search bar on [jeeshaanbakshi.vercel.app](https://jeeshaanbakshi.vercel.app) runs this same pipeline (same model) in the browser.

## Run it

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python -m rag "How does NeRF represent a 3D scene?"
```

```
61 chunks · 100 chars · 40 overlap

1. score 0.690 · chunk 10
   the core innovations in NeRF is the use of volumetric rendering, where a ray is cast from the camera

2. score 0.670 · chunk 37
   olution has been the adaptation of NeRF-like models for dynamic scenes, where the geometry and appea

3. score 0.640 · chunk 2
   nerate novel views of complex 3D scenes. Unlike traditional graphics pipelines that rely on explicit
```

Options: `--file`, `--k`, `--size`, `--overlap`. The first run downloads the model (~90 MB).

## How it works

| Step | File | What happens |
|---|---|---|
| Chunk | `rag/chunking.py` | Split the text into fixed-width windows that overlap, so an idea cut at one edge is whole in the next chunk |
| Embed | `rag/pipeline.py` | Encode every chunk in one batch with `all-MiniLM-L6-v2` (SentenceTransformers) → 384-d vectors |
| Score | `rag/retrieve.py` | Normalize, then one matrix–vector product: cosine similarity in NumPy |
| Retrieve | `rag/retrieve.py` | Sort scores, return the top-k passages |

## Tuning chunk size and overlap

`python experiments/chunk_sweep.py` — best score per question for each setting:

| size | overlap | chunks | NeRF scene | volumetric rendering | "what is this about?" |
|---:|---:|---:|---:|---:|---:|
| 60 | 20 | 92 | 0.692 | **0.779** | 0.269 |
| 100 | 0 | 37 | 0.690 | 0.671 | 0.224 |
| 100 | 40 | 61 | 0.690 | 0.724 | 0.224 |
| 200 | 50 | 25 | **0.766** | 0.641 | 0.190 |
| 400 | 100 | 12 | 0.705 | 0.662 | 0.160 |

- **Overlap helps:** at size 100, adding 40 characters of overlap raised the volumetric-rendering match from 0.671 to 0.724.
- **No single best size:** small chunks win narrow questions, bigger chunks win broad ones.
- **Vague questions score low everywhere** — small chunks can't represent a whole document.

A higher top score isn't the same as a correct answer; there are no labelled answers yet, so this is exploration, not a benchmark.

## Tests

```bash
python -m unittest discover -s tests -v
```

10 tests for chunking, cosine similarity, top-k and the pipeline. They use a tiny fake model, so they run instantly with no download.

## Limits

- Fixed-width chunks cut words in half (see the output above).
- Brute-force search over every chunk: fine for a document, not for millions.

## Next

- Pass the top passages to an LLM to write an answer grounded in them.
- Measure retrieval with labelled questions (Recall@k, MRR).
- Sentence-aware chunking.
