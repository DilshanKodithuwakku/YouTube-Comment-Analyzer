"""Repeatable topic modelling for processed YouTube comments using TF-IDF and NMF."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

LOGGER = logging.getLogger(__name__)


class TopicModelError(ValueError):
    """Raised when there is not enough meaningful text to identify topics."""


def model_topics(
    comments: pd.DataFrame,
    n_topics: int = 5,
    top_terms: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign a topic to each comment and return assignments plus topic summary.

    NMF is used deliberately: it is local, deterministic with a fixed seed, and
    does not download a model or transmit comment text to a third party.
    """
    if "cleaned_text" not in comments.columns:
        raise TopicModelError("Input must contain a cleaned_text column.")
    if n_topics < 1 or top_terms < 1:
        raise TopicModelError("n_topics and top_terms must be at least 1.")

    frame = comments.copy()
    documents = frame["cleaned_text"].fillna("").astype(str).str.strip()
    valid_mask = documents.str.len() > 0
    if valid_mask.sum() < 2:
        raise TopicModelError("At least two non-empty comments are required for topic modelling.")

    min_document_frequency = 2 if valid_mask.sum() >= 10 else 1
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=min_document_frequency,
        max_df=0.95,
        max_features=5_000,
    )
    matrix = vectorizer.fit_transform(documents[valid_mask])
    if matrix.shape[1] == 0:
        raise TopicModelError("Comments contain no usable terms after text normalization.")

    actual_topics = min(n_topics, matrix.shape[0], matrix.shape[1])
    model = NMF(
        n_components=actual_topics,
        init="nndsvda",
        random_state=42,
        max_iter=400,
    )
    weights = model.fit_transform(matrix)
    feature_names = vectorizer.get_feature_names_out()
    labels: list[str] = []
    for component in model.components_:
        term_indexes = component.argsort()[-top_terms:][::-1]
        labels.append(", ".join(feature_names[index] for index in term_indexes))

    assignments = frame.copy()
    assignments["topic_id"] = pd.NA
    assignments["topic_label"] = pd.NA
    assignments["topic_confidence"] = pd.NA
    valid_indexes = assignments.index[valid_mask]
    topic_ids = weights.argmax(axis=1)
    assignments.loc[valid_indexes, "topic_id"] = topic_ids + 1
    assignments.loc[valid_indexes, "topic_label"] = [labels[index] for index in topic_ids]
    assignments.loc[valid_indexes, "topic_confidence"] = weights.max(axis=1).round(4)

    summary = (
        assignments.loc[valid_mask]
        .groupby(["topic_id", "topic_label"], dropna=False)
        .agg(comment_count=("cleaned_text", "size"), mean_confidence=("topic_confidence", "mean"))
        .reset_index()
        .sort_values("comment_count", ascending=False)
    )
    if "sentiment_score" in assignments.columns:
        sentiment_summary = (
            assignments.loc[valid_mask]
            .groupby("topic_id", dropna=False)["sentiment_score"]
            .mean()
            .rename("mean_sentiment")
            .reset_index()
        )
        summary = summary.merge(sentiment_summary, on="topic_id", how="left")
    return assignments, summary


def run_topic_modeling(
    input_path: Path,
    assignments_output: Path,
    summary_output: Path,
    n_topics: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load processed data, model topics, and persist both useful result tables."""
    try:
        comments = pd.read_csv(input_path)
    except FileNotFoundError as error:
        raise TopicModelError(f"Input file does not exist: {input_path}") from error
    except pd.errors.EmptyDataError as error:
        raise TopicModelError(f"Input file is empty: {input_path}") from error

    assignments, summary = model_topics(comments, n_topics=n_topics)
    assignments_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(assignments_output, index=False)
    summary.to_csv(summary_output, index=False)
    return assignments, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Discover recurring YouTube comment topics.")
    parser.add_argument("--input", type=Path, default=Path("data/processed/cleaned_comments.csv"))
    parser.add_argument("--assignments-output", type=Path, default=Path("data/processed/topic_comments.csv"))
    parser.add_argument("--summary-output", type=Path, default=Path("data/processed/topic_summary.csv"))
    parser.add_argument("--topics", type=int, default=5, help="Requested number of topics (default: 5)")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        _, summary = run_topic_modeling(
            args.input, args.assignments_output, args.summary_output, args.topics
        )
    except TopicModelError as error:
        LOGGER.error("%s", error)
        return 1
    LOGGER.info("Saved %s discovered topics to %s", len(summary), args.summary_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
