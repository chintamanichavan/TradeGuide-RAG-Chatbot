# TradeGuide Knowledge Base Design

The TradeGuide knowledge base supports a retrieval-augmented chatbot for retail e-trading education. Documents are organized by topic, including market orders, limit orders, stop orders, bid-ask spreads, settlement, margin, trading halts, order execution, and terminology.

The intended workflow is:
1. A user submits a plain-language question.
2. The question is converted into an embedding.
3. The system retrieves the most relevant document passages.
4. The chatbot generates a concise educational answer.
5. The user receives the answer and the source topic.

The evaluation dataset includes sample questions and expected topics to test whether retrieval returns relevant material. TradeGuide is educational only and does not provide financial advice, trading recommendations, portfolio analysis, or price predictions.
