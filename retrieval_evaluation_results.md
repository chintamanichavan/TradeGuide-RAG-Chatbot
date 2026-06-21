# TradeGuide Retrieval Evaluation Results

- Evaluation questions: 50
- Top-1 accuracy: 96.0%
- Top-3 accuracy: 100.0%
- Embedding model: `LightweightTradingEmbeddingModel offline fallback`
- Retriever: FAISS `IndexFlatIP` over L2-normalized embeddings

## Top-1 Misses

| Question | Expected | Retrieved Top 3 |
|---|---|---|
| How is a limit order different from a market order? | limit_orders | market_orders, stop_orders, limit_orders |
| How is a limit order different from a market order? | limit_orders | market_orders, stop_orders, limit_orders |

## Recommendations

- Keep using semantic retrieval instead of keyword-only matching because it handles paraphrases better.
- Expand each knowledge-base topic with more examples; the current documents are intentionally short, which limits ranking signal.
- Add negative/out-of-scope questions to test the educational-only safety boundary.
- Re-run this evaluation after new documents or model changes.
