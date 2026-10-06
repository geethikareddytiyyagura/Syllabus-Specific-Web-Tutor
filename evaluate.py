"""Evaluate syllabus retrieval against a CSV of questions and expected keywords."""

import argparse
from pathlib import Path
import re
import sys

from pandas import DataFrame, isna, read_csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

from src.embeddings import EmbeddingModel
from src.rag_pipeline import RAGPipeline
from src.retriever import Retriever
from src.vector_store import VectorStore


def _parse_keywords(value: object) -> list[str]:
    """Split a CSV keyword cell using comma, semicolon, or vertical-bar separators."""
    if value is None or isna(value):
        return []
    return [
        keyword.strip()
        for keyword in re.split(r"[;,|]", str(value))
        if keyword.strip()
    ]


def evaluate_file(
    input_path: Path,
    output_path: Path | None = None,
    top_k: int = 5,
) -> DataFrame:
    """Run syllabus retrieval for each row and calculate hit and coverage metrics."""
    if not input_path.is_file():
        raise FileNotFoundError(f"Evaluation CSV not found: {input_path}")
    frame = read_csv(input_path)
    required = {"question", "expected_keywords"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "CSV must contain columns: question, expected_keywords. "
            f"Missing: {', '.join(sorted(missing))}"
        )
    if frame.empty:
        raise ValueError("Evaluation CSV contains no question rows.")

    pipeline = RAGPipeline(Retriever(VectorStore(), EmbeddingModel()))
    analyzer = CountVectorizer().build_analyzer()
    results: list[dict[str, object]] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    for row_number, row in frame.iterrows():
        question = str(row["question"]).strip()
        keywords = _parse_keywords(row["expected_keywords"])
        if not question or not keywords:
            raise ValueError(
                f"Row {row_number + 2} must include a question and expected keywords."
            )
        subject = (
            str(row["subject"]).strip()
            if "subject" in frame and not isna(row["subject"])
            else None
        )
        unit = (
            str(row["unit"]).strip()
            if "unit" in frame and not isna(row["unit"])
            else None
        )
        chunks = pipeline.retrieve(
            question,
            top_k=top_k,
            subject=subject or None,
            unit=unit or None,
        )
        retrieved_text = "\n".join(chunk.text for chunk in chunks)
        retrieved_folded = retrieved_text.casefold()
        hit = int(any(keyword.casefold() in retrieved_folded for keyword in keywords))
        expected_tokens = {
            token
            for keyword in keywords
            for token in analyzer(keyword.casefold())
        }
        retrieved_tokens = set(analyzer(retrieved_folded))
        coverage = (
            len(expected_tokens & retrieved_tokens) / len(expected_tokens)
            if expected_tokens
            else 0.0
        )
        y_true.append(1)
        y_pred.append(hit)
        results.append(
            {
                "question": question,
                "expected_keywords": "; ".join(keywords),
                "retrieval_hit": hit,
                "keyword_coverage": coverage,
                "retrieved_sources": "; ".join(
                    f"{chunk.metadata.get('source_file', '?')}, "
                    f"p.{chunk.metadata.get('page', '?')}"
                    for chunk in chunks
                ),
            }
        )
    report = DataFrame(results)
    hit_rate = accuracy_score(y_true, y_pred)
    mean_coverage = float(report["keyword_coverage"].mean())
    print(f"Retrieval hit rate: {hit_rate:.1%} ({sum(y_pred)}/{len(y_pred)})")
    print(f"Keyword coverage:   {mean_coverage:.1%}")
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(output_path, index=False, encoding="utf-8")
        print(f"Detailed results: {output_path}")
    return report


def main(argv: list[str] | None = None) -> int:
    """Parse evaluation options, report errors clearly, and return a process status."""
    parser = argparse.ArgumentParser(
        description="Measure retrieval hit rate and expected-keyword coverage."
    )
    parser.add_argument("csv_file", type=Path, help="CSV with question, expected_keywords")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for per-question evaluation results",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    args = parser.parse_args(argv)
    if args.top_k <= 0:
        parser.error("--top-k must be greater than zero.")
    try:
        evaluate_file(args.csv_file, args.output, args.top_k)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
