import os
import gradio as gr

DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "docs")

KEYWORDS = {
    "market_orders": ["market order", "market orders", "immediate", "best available"],
    "limit_orders": ["limit order", "limit orders", "unfilled", "limit price", "not execute"],
    "stop_orders": ["stop order", "stop price", "stop loss"],
    "bid_ask_spread": ["bid", "ask", "spread", "bid ask"],
    "settlement": ["settlement", "settle", "t+1", "trade date"],
    "margin": ["margin", "borrow", "margin call"],
    "trading_halts": ["halt", "halted", "pause", "trading paused"],
    "order_execution": ["execution", "execute", "order placed", "completed trade"],
}

def load_docs():
    docs = {}
    for filename in os.listdir(DOCS_DIR):
        if filename.endswith(".md"):
            with open(os.path.join(DOCS_DIR, filename), "r", encoding="utf-8") as f:
                docs[filename.replace(".md", "")] = f.read()
    return docs

DOCUMENTS = load_docs()

def choose_topic(question):
    q = question.lower()
    scores = {}
    for topic, words in KEYWORDS.items():
        scores[topic] = sum(1 for word in words if word in q)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "glossary"

def answer_question(question):
    topic = choose_topic(question)
    content = DOCUMENTS.get(topic, DOCUMENTS.get("glossary", ""))
    body = " ".join([p.strip() for p in content.split("\n\n") if p.strip()][1:])
    return f"{body}\n\nSource used: {topic}.md\n\nDisclaimer: TradeGuide provides general educational information only and does not provide financial advice."

demo = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(label="Ask a retail trading question"),
    outputs=gr.Textbox(label="TradeGuide Response"),
    title="TradeGuide: Retail E-Trading Education Chatbot"
)

if __name__ == "__main__":
    demo.launch()
