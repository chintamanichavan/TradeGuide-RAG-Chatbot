from pathlib import Path
import argparse

from retriever import (
    DEFAULT_MODEL_NAME,
    EmbeddingModel,
    FaissRetriever,
    LightweightTradingEmbeddingModel,
    load_markdown_chunks,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"
INDEX_DIR = PROJECT_ROOT / "index"
INDEX_PATH = INDEX_DIR / "tradeguide.faiss"
CHUNKS_PATH = INDEX_DIR / "chunks.json"


def build_index(
    model_name: str = DEFAULT_MODEL_NAME,
    offline_fallback: bool = False,
) -> tuple[int, Path, Path]:
    chunks = load_markdown_chunks(DOCS_DIR)
    embedding_model = LightweightTradingEmbeddingModel() if offline_fallback else EmbeddingModel(model_name)
    retriever = FaissRetriever(embedding_model)
    retriever.build(chunks)
    retriever.save(INDEX_PATH, CHUNKS_PATH)
    return len(chunks), INDEX_PATH, CHUNKS_PATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the TradeGuide FAISS retrieval index.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--offline-fallback",
        action="store_true",
        help="Use a small local embedding model when Hugging Face downloads are unavailable.",
    )
    args = parser.parse_args()

    chunk_count, index_path, chunks_path = build_index(
        model_name=args.model_name,
        offline_fallback=args.offline_fallback,
    )
    print(f"Built FAISS index with {chunk_count} chunks.")
    print(f"Index: {index_path}")
    print(f"Chunk metadata: {chunks_path}")
