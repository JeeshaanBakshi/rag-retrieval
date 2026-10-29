"""Try several chunk sizes and overlaps and compare what retrieval finds.

There are no labelled answers here, so this is an exploration, not a benchmark:
it prints how many chunks each setting makes and the best score per question.

Run from the project root:  python experiments/chunk_sweep.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sentence_transformers import SentenceTransformer  # noqa: E402

from rag.pipeline import MODEL_NAME, Retriever  # noqa: E402

QUESTIONS = [
    "How does NeRF represent a 3D scene?",
    "What is volumetric rendering?",
    "What is this document about?",
]
SETTINGS = [(60, 20), (100, 0), (100, 40), (200, 50), (400, 100)]

text = Path("data/nerf.txt").read_text(encoding="utf-8")
model = SentenceTransformer(MODEL_NAME)  # load once, reuse for every setting

print(f"{'size':>5} {'overlap':>7} {'chunks':>6}  " + "  ".join(f"Q{i + 1} best" for i in range(len(QUESTIONS))))
for size, overlap in SETTINGS:
    r = Retriever(text, size, overlap, model=model)
    best = [r.search(q, k=1)[0].score for q in QUESTIONS]
    print(f"{size:>5} {overlap:>7} {len(r.chunks):>6}  " + "  ".join(f"{s:>7.3f}" for s in best))

print()
for i, q in enumerate(QUESTIONS, 1):
    print(f"Q{i}: {q}")
