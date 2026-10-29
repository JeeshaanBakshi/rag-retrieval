"""A small web app for the retrieval pipeline.

    python server.py            # http://localhost:3000
    PORT=8000 python server.py  # another port

Standard library only (http.server); the model is loaded once and each
(document, chunk size, overlap) index is cached, so repeat searches are fast.
"""

import hashlib
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock

from rag.pipeline import MODEL_NAME, Retriever

ROOT = Path(__file__).resolve().parent
SAMPLE = (ROOT / "data" / "nerf.txt").read_text(encoding="utf-8")
MAX_CHARS = 50_000  # keep pasted documents small enough to embed quickly
INDEX_PAGE = (ROOT / "web" / "index.html").read_bytes()

_model = None
_cache: dict[tuple[str, int, int], Retriever] = {}
_lock = Lock()


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(MODEL_NAME)
    return _model


def search(question: str, text: str | None, k: int, size: int, overlap: int, model=None) -> dict:
    """Run one search and return a JSON-ready result. Raises ValueError on bad input."""
    question = question.strip()
    if not question:
        raise ValueError("Type a question.")
    text = (text or "").strip() or SAMPLE
    if len(text) > MAX_CHARS:
        raise ValueError(f"Document is too long (max {MAX_CHARS:,} characters).")
    if not 20 <= size <= 1000:
        raise ValueError("Chunk size must be between 20 and 1000.")
    if not 0 <= overlap < size:
        raise ValueError("Overlap must be at least 0 and smaller than the chunk size.")
    k = max(1, min(k, 10))

    model = model or get_model()
    key = (hashlib.sha1(text.encode()).hexdigest(), size, overlap)
    started = time.perf_counter()
    with _lock:  # one index build at a time; later searches reuse it
        retriever = _cache.get(key)
        if retriever is None:
            retriever = _cache[key] = Retriever(text, size, overlap, model=model)
    hits = retriever.search(question, k)
    return {
        "question": question,
        "chunks": len(retriever.chunks),
        "size": size,
        "overlap": overlap,
        "ms": round((time.perf_counter() - started) * 1000),
        "sample": text == SAMPLE,
        "hits": [{"index": h.index, "score": round(h.score, 4), "text": h.text} for h in hits],
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: bytes, kind: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, data: dict) -> None:
        self._send(status, json.dumps(data).encode(), "application/json")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, INDEX_PAGE, "text/html; charset=utf-8")
        elif self.path == "/api/sample":
            self._json(200, {"text": SAMPLE, "chars": len(SAMPLE), "model": MODEL_NAME})
        else:
            self._json(404, {"error": "Not found"})

    def do_POST(self):
        if self.path != "/api/search":
            return self._json(404, {"error": "Not found"})
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length > MAX_CHARS * 4:
                raise ValueError("Request is too large.")
            body = json.loads(self.rfile.read(length) or b"{}")
            result = search(
                str(body.get("question", "")),
                body.get("text"),
                int(body.get("k", 3)),
                int(body.get("size", 100)),
                int(body.get("overlap", 40)),
            )
            self._json(200, result)
        except (ValueError, json.JSONDecodeError) as err:
            self._json(400, {"error": str(err)})
        except Exception as err:  # model or runtime failure
            self._json(500, {"error": f"Search failed: {err}"})

    def log_message(self, fmt, *args):  # quieter console: one line per request
        print(f"{self.command} {self.path} → {args[1] if len(args) > 1 else ''}")


def main() -> None:
    port = int(os.environ.get("PORT", 3000))
    print(f"Loading {MODEL_NAME}…")
    get_model()
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"RAG retrieval running at http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
