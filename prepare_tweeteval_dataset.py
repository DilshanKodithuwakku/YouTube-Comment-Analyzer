"""Download and prepare the public TweetEval sentiment dataset for local training."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data_processor import clean_text

TWEETEVAL_DATASET = "cardiffnlp/tweet_eval"
TWEETEVAL_CONFIG = "sentiment"
LABELS = {0: "Negative", 1: "Neutral", 2: "Positive"}


class PublicDatasetError(RuntimeError):
    """Raised when the public dataset cannot be downloaded or validated."""


def prepare_tweeteval_frame(dataset: object) -> pd.DataFrame:
    """Convert TweetEval's official splits to this project's training CSV format."""
    rows: list[pd.DataFrame] = []
    for split_name in ("train", "validation", "test"):
        split = dataset[split_name]
        frame = pd.DataFrame({"cleaned_text": split["text"], "label_id": split["label"]})
        frame["actual_sentiment"] = frame["label_id"].map(LABELS)
        if frame["actual_sentiment"].isna().any():
            raise PublicDatasetError(f"TweetEval {split_name} contains an unknown sentiment label.")
        frame["cleaned_text"] = frame["cleaned_text"].map(clean_text)
        frame = frame.loc[frame["cleaned_text"].str.len() >= 3].copy()
        frame["dataset_split"] = split_name
        frame["source_dataset"] = "TweetEval sentiment"
        rows.append(frame.drop(columns="label_id"))
    return pd.concat(rows, ignore_index=True)


def download_tweeteval(output_path: Path) -> pd.DataFrame:
    """Download TweetEval from its public Hugging Face repository and save a CSV."""
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise PublicDatasetError(
            "Dataset support is not installed. Run pip install -r requirements.txt."
        ) from error
    try:
        dataset = load_dataset(TWEETEVAL_DATASET, TWEETEVAL_CONFIG)
    except Exception as error:
        raise PublicDatasetError(
            "Could not download TweetEval. Check your internet connection and Hugging Face access."
        ) from error

    frame = prepare_tweeteval_frame(dataset)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the public TweetEval sentiment dataset.")
    parser.add_argument("--output", type=Path, default=Path("data/labels/tweeteval_sentiment.csv"))
    args = parser.parse_args()
    try:
        frame = download_tweeteval(args.output)
    except PublicDatasetError as error:
        print(f"ERROR: {error}")
        return 1
    counts = frame.groupby("dataset_split").size().to_dict()
    print(f"Saved {len(frame):,} TweetEval comments to {args.output}. Splits: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
