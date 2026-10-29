"""Command line: python -m rag "your question" [--file data/nerf.txt] [--k 3] [--size 100] [--overlap 40]"""

import argparse
from pathlib import Path

from .pipeline import Retriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Find the passages that best answer a question.")
    parser.add_argument("question")
    parser.add_argument("--file", default="data/nerf.txt", help="document to search")
    parser.add_argument("--k", type=int, default=3, help="how many passages to return")
    parser.add_argument("--size", type=int, default=100, help="chunk size in characters")
    parser.add_argument("--overlap", type=int, default=40, help="characters shared by neighbouring chunks")
    args = parser.parse_args()

    text = Path(args.file).read_text(encoding="utf-8")
    retriever = Retriever(text, args.size, args.overlap)
    print(f"{len(retriever.chunks)} chunks · {args.size} chars · {args.overlap} overlap\n")
    for rank, hit in enumerate(retriever.search(args.question, args.k), 1):
        print(f"{rank}. score {hit.score:.3f} · chunk {hit.index}")
        print(f"   {hit.text.strip()}\n")


if __name__ == "__main__":
    main()
