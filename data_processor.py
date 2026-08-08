"""Clean YouTube comments and add transparent, repeatable sentiment labels."""

from __future__ import annotations

import argparse
import logging
import re
import unicodedata
from pathlib import Path
from typing import Callable

import emoji
import pandas as pd
from textblob import TextBlob

LOGGER = logging.getLogger(__name__)
REQUIRED_RAW_COLUMNS = {"text"}
SENTIMENT_ORDER = ["Positive", "Neutral", "Negative", "Uncertain"]
SentimentPredictor = Callable[[list[str]], pd.DataFrame]


class DataValidationError(ValueError):
    """Raised when an input file does not contain usable comment data."""


def clean_text(text: object) -> str:
    """Normalize comment text while retaining emoji meaning for sentiment analysis."""
    if not isinstance(text, str):
        return ""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = emoji.demojize(normalized, language="en")
    normalized = re.sub(r"https?://\S+|www\.\S+", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"(?<!\w)@\w+", "", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def get_sentiment(text: str) -> float:
    """Return TextBlob polarity as a bounded float in the range -1 to 1."""
    return float(TextBlob(text).sentiment.polarity)


def categorize_sentiment(score: float, threshold: float = 0.1) -> str:
    """Map a polarity score into one of the documented dashboard categories."""
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    if score > threshold:
        return "Positive"
    if score < -threshold:
        return "Negative"
    return "Neutral"


def process_dataframe(
    raw_comments: pd.DataFrame,
    min_text_length: int = 3,
    sentiment_threshold: float = 0.1,
    sentiment_predictor: SentimentPredictor | None = None,
) -> pd.DataFrame:
    """Return a cleaned, de-duplicated comment frame ready for analysis.

    Duplicates are removed only when a stable YouTube ``comment_id`` is present;
    distinct users posting the same text remain distinct observations.
    """
    missing = REQUIRED_RAW_COLUMNS.difference(raw_comments.columns)
    if missing:
        raise DataValidationError(f"Input is missing required column(s): {', '.join(sorted(missing))}")
    if min_text_length < 1:
        raise ValueError("min_text_length must be at least 1")

    frame = raw_comments.copy()
    initial_count = len(frame)
    if "comment_id" in frame.columns:
        frame = frame.drop_duplicates(subset="comment_id", keep="first")
    frame["cleaned_text"] = frame["text"].map(clean_text)
    frame = frame.loc[frame["cleaned_text"].str.len() >= min_text_length].copy()
    if sentiment_predictor is None:
        frame["sentiment_score"] = frame["cleaned_text"].map(get_sentiment)
        frame["sentiment_category"] = frame["sentiment_score"].map(
            lambda score: categorize_sentiment(score, sentiment_threshold)
        )
        frame["sentiment_confidence"] = pd.NA
        frame["sentiment_model"] = "textblob_baseline"
    else:
        predictions = sentiment_predictor(frame["cleaned_text"].tolist())
        required_prediction_columns = {
            "sentiment_category",
            "sentiment_score",
            "sentiment_confidence",
            "sentiment_model",
        }
        missing_prediction_columns = required_prediction_columns.difference(predictions.columns)
        if missing_prediction_columns or len(predictions) != len(frame):
            raise DataValidationError(
                "The sentiment model returned an invalid prediction result."
            )
        for column in required_prediction_columns:
            frame[column] = predictions[column].to_numpy()

    if "published_at" in frame.columns:
        frame["published_at"] = pd.to_datetime(frame["published_at"], errors="coerce", utc=True)
    if "likes" in frame.columns:
        frame["likes"] = pd.to_numeric(frame["likes"], errors="coerce").fillna(0).astype(int)

    LOGGER.info("Retained %s of %s comments after validation", len(frame), initial_count)
    return frame.reset_index(drop=True)


def process_file(
    input_path: Path,
    output_path: Path,
    min_text_length: int = 3,
    sentiment_threshold: float = 0.1,
    model_dir: Path | None = None,
    confidence_threshold: float = 0.60,
) -> pd.DataFrame:
    """Read raw CSV comments, process them, and write a processed CSV."""
    try:
        raw_comments = pd.read_csv(input_path)
    except FileNotFoundError as error:
        raise DataValidationError(f"Input file does not exist: {input_path}") from error
    except pd.errors.EmptyDataError as error:
        raise DataValidationError(f"Input file is empty: {input_path}") from error

    predictor = None
    if model_dir is not None:
        from sentiment_model import TransformerSentimentPredictor

        predictor = TransformerSentimentPredictor.load(model_dir).predict
        processed = process_dataframe(
            raw_comments,
            min_text_length,
            sentiment_threshold,
            sentiment_predictor=lambda texts: predictor(texts, confidence_threshold),
        )
    else:
        processed = process_dataframe(raw_comments, min_text_length, sentiment_threshold)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(output_path, index=False)
    return processed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean and score YouTube comments.")
    parser.add_argument("--input", type=Path, default=Path("data/raw/youtube_comments.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/cleaned_comments.csv"))
    parser.add_argument("--min-text-length", type=int, default=3)
    parser.add_argument("--sentiment-threshold", type=float, default=0.1)
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Use a fine-tuned Transformer model stored in this directory.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.60,
        help="Minimum Transformer confidence required to assign a sentiment label.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        result = process_file(
            args.input,
            args.output,
            args.min_text_length,
            args.sentiment_threshold,
            args.model_dir,
            args.confidence_threshold,
        )
    except (DataValidationError, ValueError, RuntimeError) as error:
        LOGGER.error("%s", error)
        return 1
    LOGGER.info("Saved %s processed comments to %s", len(result), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
