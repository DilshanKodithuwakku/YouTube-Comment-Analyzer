"""Prepare human labels and calculate reliable sentiment quality metrics."""

from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

SENTIMENT_LABELS = ("Negative", "Neutral", "Positive")
PREDICTION_LABELS = (*SENTIMENT_LABELS, "Uncertain")
PREDICTION_COLUMN = "sentiment_category"
GROUND_TRUTH_COLUMN = "actual_sentiment"


class LabelValidationError(ValueError):
    """Raised when a labelled evaluation file cannot be used safely."""


def normalize_sentiment_labels(
    values: Iterable[object], allowed_labels: Iterable[str] = SENTIMENT_LABELS
) -> pd.Series:
    """Normalize approved sentiment labels while preserving invalid values for validation."""
    labels = pd.Series(values, dtype="string").str.strip().str.lower()
    mapping = {label.lower(): label for label in allowed_labels}
    return labels.map(mapping)


def create_labeling_template(comments: pd.DataFrame, sample_size: int = 250) -> pd.DataFrame:
    """Return a blinded random, de-duplicated sample for independent human labels."""
    if sample_size < 1:
        raise LabelValidationError("sample_size must be at least 1.")
    required = {"cleaned_text"}
    missing = required.difference(comments.columns)
    if missing:
        raise LabelValidationError(
            f"Comments are missing required column(s): {', '.join(sorted(missing))}."
        )

    columns = [
        column
        for column in ["comment_id", "video_id", "comment_type", "text", "cleaned_text"]
        if column in comments.columns
    ]
    template = comments.loc[:, columns].drop_duplicates(subset=["cleaned_text"]).copy()
    template = template.sample(n=min(sample_size, len(template)), random_state=42)
    template[GROUND_TRUTH_COLUMN] = ""
    return template.reset_index(drop=True)


def validate_labeled_comments(
    labeled_comments: pd.DataFrame, require_predictions: bool = True
) -> pd.DataFrame:
    """Validate and normalize human labels, optionally requiring model predictions."""
    required = {GROUND_TRUTH_COLUMN}
    if require_predictions:
        required.add(PREDICTION_COLUMN)
    missing = required.difference(labeled_comments.columns)
    if missing:
        raise LabelValidationError(
            f"Labelled data is missing required column(s): {', '.join(sorted(missing))}."
        )

    frame = labeled_comments.copy()
    frame[GROUND_TRUTH_COLUMN] = normalize_sentiment_labels(frame[GROUND_TRUTH_COLUMN])
    invalid_predictions = pd.Series(False, index=frame.index)
    if require_predictions:
        frame[PREDICTION_COLUMN] = normalize_sentiment_labels(
            frame[PREDICTION_COLUMN], PREDICTION_LABELS
        )
        invalid_predictions = frame[PREDICTION_COLUMN].isna()
    invalid_labels = frame[GROUND_TRUTH_COLUMN].isna()
    if invalid_predictions.any() or invalid_labels.any():
        invalid_count = int((invalid_predictions | invalid_labels).sum())
        allowed = ", ".join(PREDICTION_LABELS if require_predictions else SENTIMENT_LABELS)
        raise LabelValidationError(
            f"{invalid_count} row(s) have missing or invalid sentiment labels. Use only: {allowed}."
        )
    if frame.empty:
        raise LabelValidationError("At least one labelled comment is required.")
    return frame


def attach_model_predictions(labeled_comments: pd.DataFrame, comments: pd.DataFrame) -> pd.DataFrame:
    """Match blinded human labels to the model predictions from the same analysis."""
    labels = validate_labeled_comments(labeled_comments, require_predictions=False)
    if PREDICTION_COLUMN not in comments.columns:
        raise LabelValidationError("The current analysis does not contain sentiment predictions.")
    match_column = "comment_id" if "comment_id" in labels and "comment_id" in comments else "cleaned_text"
    if match_column not in labels or match_column not in comments:
        raise LabelValidationError("Labels need comment_id or cleaned_text to match current predictions.")
    if labels[match_column].duplicated().any() or comments[match_column].duplicated().any():
        raise LabelValidationError(
            f"{match_column} must be unique when matching labels to current predictions."
        )
    predictions = comments.loc[:, [match_column, PREDICTION_COLUMN]]
    merged = labels.drop(columns=[PREDICTION_COLUMN], errors="ignore").merge(
        predictions, on=match_column, how="left", validate="one_to_one"
    )
    if merged[PREDICTION_COLUMN].isna().any():
        raise LabelValidationError(
            "Some labelled comments are not in the current analysis. Upload the template from this analysis."
        )
    return merged


def evaluate_sentiment_predictions(labeled_comments: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    """Calculate overall, per-label, and confusion-matrix sentiment quality metrics."""
    frame = validate_labeled_comments(labeled_comments)
    actual = frame[GROUND_TRUTH_COLUMN]
    predicted = frame[PREDICTION_COLUMN]
    report = classification_report(
        actual,
        predicted,
        labels=list(SENTIMENT_LABELS),
        output_dict=True,
        zero_division=0,
    )
    per_label = (
        pd.DataFrame(report)
        .transpose()
        .rename_axis("label")
        .reset_index()
        .query("label in @SENTIMENT_LABELS")
    )
    matrix = pd.crosstab(actual, predicted).reindex(
        index=SENTIMENT_LABELS,
        columns=PREDICTION_LABELS,
        fill_value=0,
    )
    matrix.index.name = "Actual"
    matrix.columns.name = "Predicted"
    summary = {
        "sample_size": float(len(frame)),
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, labels=list(SENTIMENT_LABELS), average="macro", zero_division=0)),
        "model_coverage": float((predicted != "Uncertain").mean()),
    }
    return summary, per_label, matrix
