from pathlib import Path
import re

import gradio as gr

from retriever import (
    EmbeddingModel,
    FaissRetriever,
    LightweightTradingEmbeddingModel,
    format_sources,
    load_markdown_chunks,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"
INDEX_PATH = PROJECT_ROOT / "index" / "tradeguide.faiss"
CHUNKS_PATH = PROJECT_ROOT / "index" / "chunks.json"

DISCLAIMER = (
    "Disclaimer: TradeGuide provides general educational information only and does not "
    "provide financial advice, trading recommendations, portfolio analysis, tax advice, "
    "legal advice, or price predictions."
)


def create_retriever() -> FaissRetriever:
    """Create the retriever using the existing TradeGuide retrieval stack."""
    try:
        embedding_model = EmbeddingModel()
    except Exception as exc:
        print(f"Sentence Transformer model unavailable; using offline fallback. Details: {exc}")
        embedding_model = LightweightTradingEmbeddingModel()

    retriever = FaissRetriever(embedding_model)

    if INDEX_PATH.exists() and CHUNKS_PATH.exists():
        retriever.load(INDEX_PATH, CHUNKS_PATH)
        index_dimension = getattr(retriever.index, "d", getattr(retriever.index, "dimension", None))

        if index_dimension != embedding_model.dimension:
            print("Index dimension mismatch. Rebuilding index with the active embedding model.")
            chunks = load_markdown_chunks(DOCS_DIR)
            retriever.build(chunks)
    else:
        print("Saved index not found. Building an in-memory index.")
        chunks = load_markdown_chunks(DOCS_DIR)
        retriever.build(chunks)

    return retriever


RETRIEVER = create_retriever()


def is_out_of_scope(question: str) -> bool:
    """
    Detect requests that move beyond education into advice, prediction,
    portfolio analysis, tax/legal advice, or trade execution.
    """
    q = question.lower().strip()

    blocked_patterns = [
        r"\bshould i buy\b",
        r"\bshould i sell\b",
        r"\bshould i hold\b",
        r"\bwhat should i buy\b",
        r"\bwhat should i sell\b",
        r"\bwhat stock should\b",
        r"\bbuy .* today\b",
        r"\bsell .* today\b",
        r"\bwill .* go up\b",
        r"\bwill .* go down\b",
        r"\bprice prediction\b",
        r"\bpredict\b",
        r"\bforecast\b",
        r"\bmy portfolio\b",
        r"\bportfolio\b",
        r"\bmy holdings\b",
        r"\bmake money\b",
        r"\bguaranteed profit\b",
        r"\bget rich\b",
        r"\btax advice\b",
        r"\blegal advice\b",
        r"\bplace a trade\b",
        r"\bexecute a trade\b",
        r"\btrade for me\b",
    ]

    return any(re.search(pattern, q) for pattern in blocked_patterns)


def safe_redirect_response() -> str:
    """Return a safe educational redirect for out-of-scope questions."""
    return (
        "TradeGuide is an educational tool and cannot provide investment advice, "
        "stock recommendations, price predictions, portfolio analysis, tax advice, "
        "legal advice, or trade execution instructions.\n\n"
        "I can still help explain general trading concepts such as market orders, "
        "limit orders, stop orders, bid-ask spread, margin, settlement, trading halts, "
        "or order execution."
    )


def generate_retrieval_grounded_answer(question: str, results) -> str:
    """
    Build a concise answer from retrieved TradeGuide content.
    """
    top_result = results[0]
    top_topic = top_result.chunk.topic
    top_title = top_result.chunk.title
    top_text = top_result.chunk.text.strip()

    answer_parts = [
        f"{top_text}",
        f"In short, this question relates most directly to {top_title}.",
    ]

    related_topics = []
    for result in results[1:]:
        topic = result.chunk.topic
        if topic != top_topic and topic not in related_topics:
            related_topics.append(topic)

    if related_topics:
        readable_related = ", ".join(related_topics[:2])
        answer_parts.append(
            f"Depending on the wording of the question, related TradeGuide topics may include "
            f"{readable_related}."
        )

    answer_parts.append(f"Source topic: {top_topic}")
    return "\n\n".join(answer_parts)


def answer_question(question: str) -> str:
    """
    Main chatbot flow:
    1. Validate the question.
    2. Block out-of-scope financial advice requests.
    3. Retrieve top knowledge-base chunks.
    4. Return a concise, source-attributed educational answer.
    """
    if not question or not question.strip():
        return "Please enter a trading education question."

    if is_out_of_scope(question):
        return f"{safe_redirect_response()}\n\n{DISCLAIMER}"

    try:
        results = RETRIEVER.search(question, top_k=3)
    except ValueError as exc:
        return f"{exc}\n\nPlease enter a trading education question."
    except Exception as exc:
        return (
            "TradeGuide could not retrieve an answer right now. Please check that the "
            "knowledge base and retrieval index are available.\n\n"
            f"Technical detail: {exc}"
        )

    if not results:
        return (
            "I could not find enough information in the TradeGuide knowledge base to answer "
            "that question. Try asking about market orders, limit orders, stop orders, "
            "bid-ask spread, settlement, margin, trading halts, or order execution."
        )

    answer = generate_retrieval_grounded_answer(question, results)
    sources = format_sources(results)

    return (
        f"{answer}\n\n"
        f"Sources used:\n{sources}\n\n"
        f"{DISCLAIMER}"
    )


def chat_response(message, history):
    """Adapter for Gradio ChatInterface."""
    return answer_question(message)


def build_demo():
    examples = [
        "What is a market order?",
        "How is a limit order different from a market order?",
        "What does bid-ask spread mean?",
        "Why can margin trading be risky?",
        "What happens during a trading halt?",
        "Why might an order not execute?",
        "Should I buy Tesla today?",
    ]

    if hasattr(gr, "ChatInterface"):
        return gr.ChatInterface(
            fn=chat_response,
            title="TradeGuide: Retail E-Trading Education Chatbot",
            description=(
                "Ask educational questions about retail trading concepts. TradeGuide uses "
                "retrieved knowledge-base content and does not provide financial advice, "
                "portfolio analysis, or price predictions."
            ),
            examples=examples,
        )

    return gr.Interface(
        fn=answer_question,
        inputs=gr.Textbox(
            label="Ask a retail trading question",
            placeholder="Example: Why might a limit order not execute?",
            lines=2,
        ),
        outputs=gr.Textbox(label="TradeGuide Response", lines=12),
        title="TradeGuide: Retail E-Trading Education Chatbot",
        description=(
            "Ask educational questions about retail trading concepts. TradeGuide uses "
            "retrieved knowledge-base content and does not provide financial advice, "
            "portfolio analysis, or price predictions."
        ),
        examples=examples,
    )


demo = build_demo()


if __name__ == "__main__":
    demo.launch()
