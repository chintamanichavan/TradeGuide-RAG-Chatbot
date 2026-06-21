from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    source: str
    topic: str
    title: str
    text: str


@dataclass(frozen=True)
class SearchResult:
    chunk: DocumentChunk
    score: float
    rank: int


class EmbeddingModel:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        embeddings = self.model.encode(
            list(texts),
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        return np.asarray(embeddings, dtype="float32")


class LightweightTradingEmbeddingModel:
    """Small offline fallback for demos when Hugging Face model download is blocked."""

    TERMS = [
        ("market_orders", ("market order", "immediate", "best available", "price change")),
        ("limit_orders", ("limit order", "limit price", "specified price", "not execute", "unfilled")),
        ("stop_orders", ("stop order", "stop price", "stop loss", "activates")),
        ("bid_ask_spread", ("bid", "ask", "spread", "buyer", "seller")),
        ("settlement", ("settlement", "settle", "t+1", "trade date", "completed")),
        ("margin", ("margin", "borrow", "borrowed", "margin call", "risky")),
        ("trading_halts", ("halt", "halted", "pause", "paused")),
        ("order_execution", ("execution", "execute", "placed", "completed trade", "guarantee")),
        ("glossary", ("term", "definition", "glossary", "means")),
    ]

    dimension = len(TERMS)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            lower = text.lower()
            vector = []
            for _topic, terms in self.TERMS:
                score = sum(1.0 for term in terms if term in lower)
                vector.append(score)
            if not any(vector):
                vector = [0.1 for _ in self.TERMS]
            vectors.append(vector)
        return np.asarray(vectors, dtype="float32")


def load_markdown_chunks(
    docs_dir: str | Path,
    max_words: int = 110,
    overlap_words: int = 25,
) -> list[DocumentChunk]:
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        raise FileNotFoundError(f"Docs directory not found: {docs_path}")

    chunks: list[DocumentChunk] = []
    for file_path in sorted(docs_path.glob("*.md")):
        content = file_path.read_text(encoding="utf-8")
        title, body = _split_title_and_body(content, file_path.stem)
        topic = file_path.stem
        parts = _chunk_text(body, max_words=max_words, overlap_words=overlap_words)
        for index, part in enumerate(parts):
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{topic}-{index:03d}",
                    source=file_path.name,
                    topic=topic,
                    title=title,
                    text=part,
                )
            )

    if not chunks:
        raise ValueError(f"No markdown documents found in {docs_path}")
    return chunks


class FaissRetriever:
    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model
        self.index: faiss.Index | None = None
        self.chunks: list[DocumentChunk] = []

    def build(self, chunks: Sequence[DocumentChunk]) -> None:
        if not chunks:
            raise ValueError("Cannot build a FAISS index with no chunks.")

        embeddings = self._embed([chunk.text for chunk in chunks])
        dimension = embeddings.shape[1]
        index = _create_index(dimension)
        index.add(embeddings)
        self.index = index
        self.chunks = list(chunks)

    def search(self, question: str, top_k: int = 3) -> list[SearchResult]:
        if not question or not question.strip():
            raise ValueError("A non-empty question is required for retrieval.")
        if self.index is None:
            raise RuntimeError("FAISS index is not loaded. Build or load an index first.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query = self._embed([question])
        count = min(top_k, len(self.chunks))
        scores, indices = self.index.search(query, count)
        results: list[SearchResult] = []
        for rank, (score, chunk_index) in enumerate(zip(scores[0], indices[0]), start=1):
            if chunk_index < 0:
                continue
            results.append(
                SearchResult(
                    chunk=self.chunks[int(chunk_index)],
                    score=float(score),
                    rank=rank,
                )
            )
        return results

    def save(self, index_path: str | Path, chunks_path: str | Path) -> None:
        if self.index is None:
            raise RuntimeError("Cannot save before building or loading an index.")
        index_file = Path(index_path)
        chunks_file = Path(chunks_path)
        index_file.parent.mkdir(parents=True, exist_ok=True)
        chunks_file.parent.mkdir(parents=True, exist_ok=True)
        _write_index(self.index, index_file)
        chunks_file.write_text(
            json.dumps([asdict(chunk) for chunk in self.chunks], indent=2),
            encoding="utf-8",
        )

    def load(self, index_path: str | Path, chunks_path: str | Path) -> None:
        index_file = Path(index_path)
        chunks_file = Path(chunks_path)
        if not index_file.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_file}")
        if not chunks_file.exists():
            raise FileNotFoundError(f"Chunk metadata not found: {chunks_file}")

        self.index = _read_index(index_file)
        data = json.loads(chunks_file.read_text(encoding="utf-8"))
        self.chunks = [DocumentChunk(**item) for item in data]

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        embeddings = self.embedding_model.encode(texts)
        embeddings = np.asarray(embeddings, dtype="float32")
        if embeddings.ndim != 2:
            raise ValueError("Embedding model must return a 2D array.")
        _normalize_l2(embeddings)
        return embeddings


class _NumpyIndexFlatIP:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.vectors = np.empty((0, dimension), dtype="float32")

    def add(self, embeddings: np.ndarray) -> None:
        if embeddings.shape[1] != self.dimension:
            raise ValueError("Embedding dimension does not match index dimension.")
        self.vectors = np.vstack([self.vectors, embeddings.astype("float32")])

    def search(self, query: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        scores = query @ self.vectors.T
        order = np.argsort(-scores, axis=1)[:, :top_k]
        sorted_scores = np.take_along_axis(scores, order, axis=1)
        return sorted_scores.astype("float32"), order.astype("int64")


def _create_index(dimension: int):
    if faiss is not None:
        return faiss.IndexFlatIP(dimension)
    return _NumpyIndexFlatIP(dimension)


def _normalize_l2(embeddings: np.ndarray) -> None:
    if faiss is not None:
        faiss.normalize_L2(embeddings)
        return
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings /= norms


def _write_index(index, index_path: Path) -> None:
    if faiss is not None:
        faiss.write_index(index, str(index_path))
        return
    payload = {
        "dimension": index.dimension,
        "vectors": index.vectors.tolist(),
    }
    index_path.write_text(json.dumps(payload), encoding="utf-8")


def _read_index(index_path: Path):
    if faiss is not None:
        return faiss.read_index(str(index_path))
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    index = _NumpyIndexFlatIP(int(payload["dimension"]))
    index.vectors = np.asarray(payload["vectors"], dtype="float32")
    return index


def _split_title_and_body(content: str, fallback_title: str) -> tuple[str, str]:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        return fallback_title.replace("_", " ").title(), ""
    first = lines[0]
    if first.startswith("#"):
        title = first.lstrip("#").strip()
        body = "\n".join(lines[1:]).strip()
        return title, body
    return fallback_title.replace("_", " ").title(), "\n".join(lines)


def _chunk_text(text: str, max_words: int, overlap_words: int) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if paragraphs and all(_word_count(paragraph) <= max_words for paragraph in paragraphs):
        return [re.sub(r"\s+", " ", paragraph).strip() for paragraph in paragraphs]

    words = cleaned.split()
    if len(words) <= max_words:
        return [cleaned]

    chunks = []
    step = max(1, max_words - overlap_words)
    for start in range(0, len(words), step):
        window = words[start : start + max_words]
        if window:
            chunks.append(" ".join(window))
        if start + max_words >= len(words):
            break
    return chunks


def _word_count(text: str) -> int:
    return len(text.split())


def format_sources(results: Iterable[SearchResult]) -> str:
    lines = []
    for result in results:
        lines.append(
            f"{result.rank}. {result.chunk.source} "
            f"(chunk {result.chunk.chunk_id}, score {result.score:.3f})"
        )
    return "\n".join(lines)
