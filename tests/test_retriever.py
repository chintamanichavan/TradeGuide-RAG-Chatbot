import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from retriever import FaissRetriever, load_markdown_chunks


class FakeEmbeddingModel:
    dimension = 3

    def encode(self, texts):
        vectors = []
        for text in texts:
            lower = text.lower()
            if "market order" in lower or "immediately" in lower:
                vectors.append([1.0, 0.0, 0.0])
            elif "limit order" in lower or "specified price" in lower:
                vectors.append([0.0, 1.0, 0.0])
            elif "margin" in lower:
                vectors.append([0.0, 0.0, 1.0])
            else:
                vectors.append([0.2, 0.2, 0.2])
        return np.array(vectors, dtype="float32")


class RetrieverTests(unittest.TestCase):
    def test_load_markdown_chunks_includes_metadata(self):
        docs_dir = PROJECT_ROOT / "docs"

        chunks = load_markdown_chunks(docs_dir)

        self.assertTrue(chunks)
        first = chunks[0]
        self.assertTrue(first.chunk_id)
        self.assertTrue(first.source.endswith(".md"))
        self.assertTrue(first.topic)
        self.assertTrue(first.text.strip())

    def test_faiss_retriever_returns_expected_topic_for_fake_embeddings(self):
        docs_dir = PROJECT_ROOT / "docs"
        chunks = load_markdown_chunks(docs_dir)
        retriever = FaissRetriever(FakeEmbeddingModel())
        retriever.build(chunks)

        results = retriever.search("What is a limit order?", top_k=3)

        self.assertTrue(results)
        self.assertEqual(results[0].chunk.topic, "limit_orders")
        self.assertGreater(results[0].score, 0)

    def test_faiss_retriever_can_save_and_load_index(self):
        docs_dir = PROJECT_ROOT / "docs"
        chunks = load_markdown_chunks(docs_dir)
        retriever = FaissRetriever(FakeEmbeddingModel())
        retriever.build(chunks)

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "tradeguide.faiss"
            chunks_path = Path(tmpdir) / "chunks.json"
            retriever.save(index_path, chunks_path)

            loaded = FaissRetriever(FakeEmbeddingModel())
            loaded.load(index_path, chunks_path)
            results = loaded.search("What is margin trading?", top_k=1)

        self.assertEqual(results[0].chunk.topic, "margin")

    def test_search_rejects_blank_question(self):
        docs_dir = PROJECT_ROOT / "docs"
        chunks = load_markdown_chunks(docs_dir)
        retriever = FaissRetriever(FakeEmbeddingModel())
        retriever.build(chunks)

        with self.assertRaisesRegex(ValueError, "question"):
            retriever.search("   ")


if __name__ == "__main__":
    unittest.main()
