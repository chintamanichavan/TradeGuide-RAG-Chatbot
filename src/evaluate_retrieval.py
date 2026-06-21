from __future__ import annotations

import csv
import argparse
from dataclasses import dataclass
from pathlib import Path

from retriever import (
    DEFAULT_MODEL_NAME,
    EmbeddingModel,
    FaissRetriever,
    LightweightTradingEmbeddingModel,
    load_markdown_chunks,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"
QUESTIONS_PATH = PROJECT_ROOT / "test_questions" / "tradeguide_test_questions.csv"
RESULTS_PATH = PROJECT_ROOT / "retrieval_evaluation_results.md"


@dataclass(frozen=True)
class EvaluationResult:
    question: str
    expected_topic: str
    retrieved_topics: list[str]
    top_1_match: bool
    top_3_match: bool


def evaluate(
    top_k: int = 3,
    model_name: str = DEFAULT_MODEL_NAME,
    offline_fallback: bool = False,
) -> list[EvaluationResult]:
    chunks = load_markdown_chunks(DOCS_DIR)
    embedding_model = LightweightTradingEmbeddingModel() if offline_fallback else EmbeddingModel(model_name)
    retriever = FaissRetriever(embedding_model)
    retriever.build(chunks)

    rows = _load_questions(QUESTIONS_PATH)
    results = []
    for row in rows:
        retrieved = retriever.search(row["question"], top_k=top_k)
        topics = [result.chunk.topic for result in retrieved]
        results.append(
            EvaluationResult(
                question=row["question"],
                expected_topic=row["expected_topic"],
                retrieved_topics=topics,
                top_1_match=bool(topics and topics[0] == row["expected_topic"]),
                top_3_match=row["expected_topic"] in topics[:3],
            )
        )
    return results


def summarize(results: list[EvaluationResult]) -> dict[str, float]:
    if not results:
        return {"count": 0, "top_1_accuracy": 0.0, "top_3_accuracy": 0.0}
    return {
        "count": float(len(results)),
        "top_1_accuracy": sum(item.top_1_match for item in results) / len(results),
        "top_3_accuracy": sum(item.top_3_match for item in results) / len(results),
    }


def write_report(
    results: list[EvaluationResult],
    output_path: Path = RESULTS_PATH,
    model_label: str = DEFAULT_MODEL_NAME,
) -> None:
    summary = summarize(results)
    misses = [item for item in results if not item.top_1_match]
    lines = [
        "# TradeGuide Retrieval Evaluation Results",
        "",
        f"- Evaluation questions: {int(summary['count'])}",
        f"- Top-1 accuracy: {summary['top_1_accuracy']:.1%}",
        f"- Top-3 accuracy: {summary['top_3_accuracy']:.1%}",
        f"- Embedding model: `{model_label}`",
        f"- Retriever: FAISS `IndexFlatIP` over L2-normalized embeddings",
        "",
        "## Top-1 Misses",
        "",
    ]
    if misses:
        lines.append("| Question | Expected | Retrieved Top 3 |")
        lines.append("|---|---|---|")
        for item in misses:
            retrieved = ", ".join(item.retrieved_topics)
            lines.append(f"| {item.question} | {item.expected_topic} | {retrieved} |")
    else:
        lines.append("No top-1 misses on the supplied evaluation dataset.")

    lines.extend(
        [
            "",
            "## Recommendations",
            "",
            "- Keep using semantic retrieval instead of keyword-only matching because it handles paraphrases better.",
            "- Expand each knowledge-base topic with more examples; the current documents are intentionally short, which limits ranking signal.",
            "- Add negative/out-of-scope questions to test the educational-only safety boundary.",
            "- Re-run this evaluation after new documents or model changes.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _load_questions(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TradeGuide retrieval accuracy.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--offline-fallback",
        action="store_true",
        help="Use a small local embedding model when Hugging Face downloads are unavailable.",
    )
    args = parser.parse_args()

    model_label = (
        "LightweightTradingEmbeddingModel offline fallback"
        if args.offline_fallback
        else args.model_name
    )
    evaluation_results = evaluate(
        model_name=args.model_name,
        offline_fallback=args.offline_fallback,
    )
    metrics = summarize(evaluation_results)
    write_report(evaluation_results, model_label=model_label)
    print(f"Questions: {int(metrics['count'])}")
    print(f"Top-1 accuracy: {metrics['top_1_accuracy']:.1%}")
    print(f"Top-3 accuracy: {metrics['top_3_accuracy']:.1%}")
    print(f"Report written to: {RESULTS_PATH}")
