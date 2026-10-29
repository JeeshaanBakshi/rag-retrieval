"""Unit tests. They use a tiny fake embedding model, so no download is needed."""

import unittest

import numpy as np

from rag import Retriever, chunk_text, cosine_scores, top_k


class FakeModel:
    """Embeds text as letter counts for a, b, c — enough to test the pipeline wiring."""

    def encode(self, text):
        if isinstance(text, list):
            return np.array([self.encode(t) for t in text])
        return np.array([text.count("a"), text.count("b"), text.count("c")], dtype=float) + 1e-9


class TestChunking(unittest.TestCase):
    def test_windows_overlap(self):
        self.assertEqual(chunk_text("abcdefghij", size=4, overlap=2), ["abcd", "cdef", "efgh", "ghij"])

    def test_no_overlap(self):
        self.assertEqual(chunk_text("abcdef", size=2, overlap=0), ["ab", "cd", "ef"])

    def test_matches_original_prototype(self):
        # main.py: 100-char chunks, 40 overlap, 3,696-char document → 61 chunks
        self.assertEqual(len(chunk_text("x" * 3696, 100, 40)), 61)

    def test_every_character_is_covered(self):
        text = "".join(chr(97 + i % 26) for i in range(500))
        chunks = chunk_text(text, 50, 10)
        covered = set()
        for i, c in enumerate(chunks):
            start = 0 if i == 0 else 50 + (i - 1) * 40 - 10
            covered.update(range(start, start + len(c)))
        self.assertEqual(covered, set(range(len(text))))

    def test_rejects_bad_settings(self):
        with self.assertRaises(ValueError):
            chunk_text("abc", size=0)
        with self.assertRaises(ValueError):
            chunk_text("abc", size=5, overlap=5)


class TestRetrieval(unittest.TestCase):
    def test_cosine_scores(self):
        chunks = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        scores = cosine_scores(np.array([1.0, 0.0]), chunks)
        np.testing.assert_allclose(scores, [1.0, 0.0, 1 / np.sqrt(2)])

    def test_scale_does_not_matter(self):
        chunks = np.array([[3.0, 4.0]])
        self.assertAlmostEqual(cosine_scores(np.array([30.0, 40.0]), chunks)[0], 1.0)

    def test_top_k_best_first(self):
        self.assertEqual(top_k(np.array([0.1, 0.9, 0.5, 0.7]), k=3), [1, 3, 2])

    def test_top_k_larger_than_corpus(self):
        self.assertEqual(top_k(np.array([0.2, 0.1]), k=5), [0, 1])

    def test_pipeline_end_to_end(self):
        r = Retriever("aaaa" + "bbbb" + "cccc", size=4, overlap=0, model=FakeModel())
        self.assertEqual(r.chunks, ["aaaa", "bbbb", "cccc"])
        hits = r.search("b", k=1)
        self.assertEqual(hits[0].text, "bbbb")
        self.assertAlmostEqual(hits[0].score, 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
