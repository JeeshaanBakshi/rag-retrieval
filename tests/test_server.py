"""Tests for the web app's search function, with the fake model (no download)."""

import unittest

import server
from tests.test_rag import FakeModel


class TestServerSearch(unittest.TestCase):
    def test_returns_ranked_hits(self):
        text = "a" * 20 + "b" * 20 + "c" * 20
        out = server.search("b", text, k=2, size=20, overlap=0, model=FakeModel())
        self.assertEqual(out["chunks"], 3)
        self.assertEqual(out["hits"][0]["text"], "b" * 20)
        self.assertEqual(len(out["hits"]), 2)
        self.assertFalse(out["sample"])

    def test_uses_sample_when_no_text(self):
        out = server.search("NeRF", None, k=1, size=100, overlap=40, model=FakeModel())
        self.assertTrue(out["sample"])
        self.assertEqual(out["chunks"], 61)

    def test_rejects_bad_input(self):
        for args in [("", None, 3, 100, 40), ("q", None, 3, 100, 100), ("q", None, 3, 5, 0), ("q", "x" * 60_000, 3, 100, 40)]:
            with self.assertRaises(ValueError):
                server.search(*args, model=FakeModel())

    def test_index_is_cached(self):
        model = FakeModel()
        server.search("a", "abcabcabc", 1, 20, 5, model=model)
        before = len(server._cache)
        server.search("b", "abcabcabc", 1, 20, 5, model=model)
        self.assertEqual(len(server._cache), before)


if __name__ == "__main__":
    unittest.main()
