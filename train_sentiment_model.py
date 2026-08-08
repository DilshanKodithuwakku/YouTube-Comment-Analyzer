"""Fine-tune and evaluate a local multilingual Transformer sentiment model."""

from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from sentiment_evaluation import (
    GROUND_TRUTH_COLUMN,
    SENTIMENT_LABELS,
    LabelValidationError,
    evaluate_sentiment_predictions,
    validate_labeled_comments,
)

DEFAULT_MODEL = "FacebookAI/xlm-roberta-base"
CPU_FAST_MODEL = "distilbert-base-uncased"


@dataclass
class DatasetSplit:
    """Train, validation, and untouched test records."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


class TrainingDataError(ValueError):
    """Raised when labels cannot create reliable training splits."""


class EncodedCommentDataset:
    """Minimal PyTorch-compatible dataset used by Hugging Face Trainer."""

    def __init__(self, encodings: dict[str, list[int]], labels: list[int]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, object]:
        import torch

        item = {name: torch.tensor(values[index]) for name, values in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index])
        return item


def _ensure_all_labels(frame: pd.DataFrame, split_name: str) -> None:
    missing = set(SENTIMENT_LABELS).difference(frame[GROUND_TRUTH_COLUMN].unique())
    if missing:
        raise TrainingDataError(
            f"The {split_name} split is missing label(s): {', '.join(sorted(missing))}. "
            "Add more human-labelled comments for every class."
        )


def split_labeled_comments(labeled_comments: pd.DataFrame) -> DatasetSplit:
    """Split labels into 70/15/15 partitions without video leakage when possible."""
    frame = validate_labeled_comments(labeled_comments, require_predictions=False).reset_index(drop=True)
    if len(frame) < 60:
        raise TrainingDataError("Use at least 60 labelled comments before training a model.")
    _ensure_all_labels(frame, "full dataset")

    if "dataset_split" in frame.columns:
        expected_splits = {"train", "validation", "test"}
        present_splits = set(frame["dataset_split"].dropna().astype(str))
        if present_splits != expected_splits:
            raise TrainingDataError(
                "dataset_split must contain exactly train, validation, and test when provided."
            )
        train = frame.loc[frame["dataset_split"] == "train"].reset_index(drop=True)
        validation = frame.loc[frame["dataset_split"] == "validation"].reset_index(drop=True)
        test = frame.loc[frame["dataset_split"] == "test"].reset_index(drop=True)
    elif "video_id" in frame.columns and frame["video_id"].nunique() >= 3:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
        train_validation_indexes, test_indexes = next(
            splitter.split(frame, groups=frame["video_id"])
        )
        train_validation = frame.iloc[train_validation_indexes].reset_index(drop=True)
        test = frame.iloc[test_indexes].reset_index(drop=True)
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=43)
        train_indexes, validation_indexes = next(
            splitter.split(train_validation, groups=train_validation["video_id"])
        )
        train = train_validation.iloc[train_indexes].reset_index(drop=True)
        validation = train_validation.iloc[validation_indexes].reset_index(drop=True)
    else:
        train_validation, test = train_test_split(
            frame,
            test_size=0.15,
            random_state=42,
            stratify=frame[GROUND_TRUTH_COLUMN],
        )
        train, validation = train_test_split(
            train_validation,
            test_size=0.1765,
            random_state=43,
            stratify=train_validation[GROUND_TRUTH_COLUMN],
        )
        train, validation, test = (part.reset_index(drop=True) for part in (train, validation, test))

    for split_name, split in (("training", train), ("validation", validation), ("test", test)):
        _ensure_all_labels(split, split_name)
    return DatasetSplit(train=train, validation=validation, test=test)


def train_model(
    input_path: Path,
    output_dir: Path,
    model_name: str = DEFAULT_MODEL,
    epochs: int = 3,
    batch_size: int = 8,
    max_length: int = 256,
    checkpoint_steps: int = 500,
) -> dict[str, float]:
    """Fine-tune a multilingual classifier and save its untouched test metrics."""
    if epochs < 1 or batch_size < 1 or max_length < 8 or checkpoint_steps < 1:
        raise TrainingDataError(
            "epochs, batch_size, and checkpoint_steps must be at least 1; max_length must be at least 8."
        )
    try:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
        from transformers.trainer_utils import get_last_checkpoint
    except ImportError as error:
        raise TrainingDataError(
            "Transformer training dependencies are missing. Run pip install -r requirements.txt."
        ) from error

    try:
        labeled_comments = pd.read_csv(input_path)
    except FileNotFoundError as error:
        raise TrainingDataError(f"Labelled data file does not exist: {input_path}") from error
    except pd.errors.EmptyDataError as error:
        raise TrainingDataError(f"Labelled data file is empty: {input_path}") from error

    if "cleaned_text" not in labeled_comments.columns:
        raise TrainingDataError("Labelled data must contain a cleaned_text column.")
    try:
        splits = split_labeled_comments(labeled_comments)
    except LabelValidationError as error:
        raise TrainingDataError(str(error)) from error

    label_to_id = {label: index for index, label in enumerate(SENTIMENT_LABELS)}
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode(frame: pd.DataFrame) -> EncodedCommentDataset:
        encodings = tokenizer(
            frame["cleaned_text"].astype(str).tolist(), truncation=True, max_length=max_length
        )
        labels = frame[GROUND_TRUTH_COLUMN].map(label_to_id).astype(int).tolist()
        return EncodedCommentDataset(encodings, labels)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(SENTIMENT_LABELS),
        id2label={index: label for label, index in label_to_id.items()},
        label2id=label_to_id,
    )

    def compute_metrics(prediction) -> dict[str, float]:
        predicted_ids = np.argmax(prediction.predictions, axis=1)
        frame = pd.DataFrame(
            {
                "sentiment_category": [SENTIMENT_LABELS[index] for index in predicted_ids],
                GROUND_TRUTH_COLUMN: [SENTIMENT_LABELS[index] for index in prediction.label_ids],
            }
        )
        summary, _, _ = evaluate_sentiment_predictions(frame)
        return summary

    output_dir.mkdir(parents=True, exist_ok=True)
    training_options = {
        "output_dir": str(output_dir),
        "learning_rate": 2e-5,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "num_train_epochs": epochs,
        "save_strategy": "steps",
        "save_steps": checkpoint_steps,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "report_to": [],
        "dataloader_pin_memory": torch.cuda.is_available(),
        "logging_steps": min(25, checkpoint_steps),
        "seed": 42,
    }
    evaluation_option = (
        "eval_strategy"
        if "eval_strategy" in inspect.signature(TrainingArguments).parameters
        else "evaluation_strategy"
    )
    training_options[evaluation_option] = "steps"
    training_options["eval_steps"] = checkpoint_steps
    training_args = TrainingArguments(**training_options)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encode(splits.train),
        eval_dataset=encode(splits.validation),
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    last_checkpoint = get_last_checkpoint(str(output_dir))
    if last_checkpoint:
        print(f"Resuming training from checkpoint: {last_checkpoint}")
    trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    test_prediction = trainer.predict(encode(splits.test))
    test_labels = [SENTIMENT_LABELS[index] for index in test_prediction.label_ids]
    test_predictions = [SENTIMENT_LABELS[index] for index in np.argmax(test_prediction.predictions, axis=1)]
    evaluation_frame = splits.test.copy()
    evaluation_frame["sentiment_category"] = test_predictions
    evaluation_frame[GROUND_TRUTH_COLUMN] = test_labels
    summary, _, _ = evaluate_sentiment_predictions(evaluation_frame)
    evaluation_frame.to_csv(output_dir / "test_predictions.csv", index=False)
    (output_dir / "test_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a multilingual YouTube-comment sentiment model.")
    parser.add_argument("--input", type=Path, required=True, help="Human-labelled CSV with actual_sentiment")
    parser.add_argument("--output-dir", type=Path, default=Path("models/sentiment"))
    parser.add_argument(
        "--model",
        default=None,
        help=f"Base model (default: {DEFAULT_MODEL}; --cpu-fast default: {CPU_FAST_MODEL})",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs (default: 3; --cpu-fast: 1)")
    parser.add_argument("--batch-size", type=int, default=None, help="Per-device batch size (default: 8; --cpu-fast: 16)")
    parser.add_argument("--max-length", type=int, default=None, help="Maximum tokens per comment (default: 256; --cpu-fast: 128)")
    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        default=500,
        help="Save a resumable checkpoint and evaluate every N training steps (default: 500)",
    )
    parser.add_argument(
        "--cpu-fast",
        action="store_true",
        help="Use a smaller English model and one epoch for a quicker CPU-only first model.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.cpu_fast:
        model_name = args.model or CPU_FAST_MODEL
        epochs = args.epochs if args.epochs is not None else 1
        batch_size = args.batch_size if args.batch_size is not None else 16
        max_length = args.max_length if args.max_length is not None else 128
    else:
        model_name = args.model or DEFAULT_MODEL
        epochs = args.epochs if args.epochs is not None else 3
        batch_size = args.batch_size if args.batch_size is not None else 8
        max_length = args.max_length if args.max_length is not None else 256
    try:
        metrics = train_model(
            args.input,
            args.output_dir,
            model_name,
            epochs,
            batch_size,
            max_length,
            args.checkpoint_steps,
        )
    except TrainingDataError as error:
        print(f"ERROR: {error}")
        return 1
    print(
        f"Saved model to {args.output_dir}. Test accuracy: {metrics['accuracy']:.1%}; "
        f"macro F1: {metrics['macro_f1']:.1%}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
