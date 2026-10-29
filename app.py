"""RAG retrieval — my original pipeline (chunk → embed → cosine → top 3), served as a small website.

Run:  python app.py   →   http://localhost:3000
"""

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

HERE = Path(__file__).resolve().parent


# STEP 1: Read the document
with open(HERE / "text.txt", "r", encoding="utf-8") as file:
    content = file.read()


# STEP 2: Split the document into overlapping chunks
def make_chunks(content, size=100, overlap=40):
    cs = [content[0:size]]

    for i in range(size, len(content), size - overlap):
        start = i - overlap
        end = i + size - overlap
        cs.append(content[start:end])

    return cs


# STEP 3: Load the embedding model (once, when the server starts)
# Similar meanings should have similar vector representations.
model = SentenceTransformer("all-MiniLM-L6-v2")


# STEP 4: Generate embeddings for every document chunk
def make_embeddings(cs):
    chunk_emb = []

    for chunk in cs:
        chunk_emb.append(model.encode(chunk))

    return chunk_emb


# STEP 5: Find the most relevant chunks for a question
def search(question, cs, chunk_emb):
    qem = model.encode(question).reshape(1, -1)

    scores = []

    for embedding in chunk_emb:
        score = cosine_similarity(qem, embedding.reshape(1, -1))[0][0]
        scores.append(float(score))

    # Create indices for all scores
    topscindex = list(range(len(scores)))

    # Sort indices by their scores, highest first
    for i in range(len(topscindex)):
        for j in range(i + 1, len(topscindex)):
            if scores[topscindex[j]] > scores[topscindex[i]]:
                topscindex[i], topscindex[j] = topscindex[j], topscindex[i]

    # Keep only the top 3 indices
    topscindex = topscindex[:3]

    # Retrieve the corresponding chunks
    highl = []

    for i in topscindex:
        highl.append({"index": i, "score": round(scores[i], 3), "text": cs[i]})

    return highl


# Chunks and embeddings are reused for the same size/overlap, so only the question is embedded per search.
cache = {}


def run(question, size, overlap):
    if (size, overlap) not in cache:
        cs = make_chunks(content, size, overlap)
        cache[(size, overlap)] = (cs, make_embeddings(cs))
    cs, chunk_emb = cache[(size, overlap)]

    return {
        # the same numbers main.py printed
        "log": {
            "total_chunks": len(cs),
            "last_chunk_length": len(cs[-1]),
            "embedding_shape": list(chunk_emb[0].shape),
            "embedding_dimensions": chunk_emb[0].ndim,
        },
        "highl": search(question, cs, chunk_emb),
    }


# ─── Website ───────────────────────────────────────────────


class Handler(BaseHTTPRequestHandler):
    def send(self, status, body, kind):
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, status, data):
        self.send(status, json.dumps(data).encode(), "application/json")

    def do_GET(self):
        if self.path == "/":
            self.send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/document":
            self.send_json(200, {"characters": len(content), "preview": content[:400]})
        else:
            self.send_json(404, {"error": "Not found"})

    def do_POST(self):
        if self.path != "/api/search":
            return self.send_json(404, {"error": "Not found"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            question = str(body.get("question", "")).strip()
            size = int(body.get("size", 100))
            overlap = int(body.get("overlap", 40))
            if not question:
                raise ValueError("Type a question first.")
            if not 20 <= size <= 500 or not 0 <= overlap < size:
                raise ValueError("Chunk size must be 20–500 and overlap smaller than it.")
            self.send_json(200, run(question, size, overlap))
        except ValueError as err:
            self.send_json(400, {"error": str(err)})

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    print(f"RAG demo running at http://localhost:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
