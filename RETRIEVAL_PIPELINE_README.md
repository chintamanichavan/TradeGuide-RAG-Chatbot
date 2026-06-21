# TradeGuide RAG Retrieval Pipeline Contribution

This folder upgrades the original keyword-based TradeGuide prototype with a runnable retrieval pipeline for the group RAG architecture.

## What This Contribution Covers

- Sentence Transformer embeddings with `sentence-transformers/all-MiniLM-L6-v2`
- Markdown knowledge-base preprocessing and chunking
- FAISS vector indexing and persistence
- Semantic top-k retrieval for user questions
- Source attribution for retrieved chunks
- Evaluation against the supplied `test_questions/tradeguide_test_questions.csv`
- Unit tests for chunking, indexing, retrieval, persistence, and blank-query handling

## Why `all-MiniLM-L6-v2`

`sentence-transformers/all-MiniLM-L6-v2` is a good fit for this project because it is small, widely used for semantic search, fast enough on CPU, and produces 384-dimensional embeddings that work well with FAISS for a lightweight student prototype.

## File Guide

- `src/retriever.py`: Core retrieval module. Loads documents, chunks text, creates embeddings, builds/loads/saves FAISS indexes, and runs top-k search.
- `src/build_index.py`: Builds `index/tradeguide.faiss` and `index/chunks.json`.
- `src/evaluate_retrieval.py`: Runs the evaluation CSV and writes `retrieval_evaluation_results.md`.
- `src/app.py`: Gradio app updated to use semantic retrieval.
- `tests/test_retriever.py`: Unit tests using a deterministic fake embedding model.
- `retrieval_evaluation_results.md`: Retrieval metrics and recommendations.

## Setup

From this folder:

```bash
pip install -r src/requirements.txt
```

## Build the FAISS Index

```bash
python src/build_index.py
```

This creates:

- `index/tradeguide.faiss`
- `index/chunks.json`

The app can also build an in-memory index automatically if these files are missing.

If Hugging Face model download is blocked by SSL/certificate policy, use the local demo fallback:

```bash
python src/build_index.py --offline-fallback
```

The fallback is only for locked-down/offline demos. The intended RAG retrieval model remains `sentence-transformers/all-MiniLM-L6-v2`.

## Run the App

```bash
python src/app.py
```

## Run Tests

```bash
python -m pytest tests -v
```

The unit tests use a fake embedding model, so they do not require downloading the Sentence Transformer model.

## Run Retrieval Evaluation

```bash
python src/evaluate_retrieval.py
```

This evaluates whether the expected topic appears in the top retrieved chunks.

Offline fallback evaluation:

```bash
python src/evaluate_retrieval.py --offline-fallback
```

## Retrieval Workflow

1. Markdown files in `docs/` are loaded.
2. Each document is split into paragraph-sized chunks with metadata.
3. Chunks are embedded with Sentence Transformers.
4. Embeddings are L2-normalized.
5. FAISS `IndexFlatIP` stores the normalized vectors.
6. User questions are embedded and normalized the same way.
7. FAISS returns the highest-scoring chunks.
8. The app displays retrieved educational content and source citations.

## Recommendations for the Group

- Expand the knowledge base with more examples per topic; very short documents reduce semantic ranking signal.
- Add out-of-scope evaluation questions such as investment advice, price predictions, or portfolio recommendations.
- Keep the retrieval pipeline separate from the language model so retrieval can be tested independently.
- Re-run `src/evaluate_retrieval.py` after any document, model, or chunking changes.
