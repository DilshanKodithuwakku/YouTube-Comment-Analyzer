import pandas as pd
import pytest

from sentiment_evaluation import (
    LabelValidationError,
    attach_model_predictions,
    create_labeling_template,
    evaluate_sentiment_predictions,
)
from train_sentiment_model import TrainingDataError, split_labeled_comments
from prepare_tweeteval_dataset import prepare_tweeteval_frame


def test_label_template_hides_predictions_and_adds_empty_human_label():
    comments = pd.DataFrame(
        {
            "comment_id": ["a", "b", "c"],
            "cleaned_text": ["great video", "bad sound", "okay"],
            "sentiment_category": ["Positive", "Negative", "Neutral"],
        }
    )

    template = create_labeling_template(comments, sample_size=2)

    assert len(template) == 2
    assert set(template.columns) == {"comment_id", "cleaned_text", "actual_sentiment"}
    assert template["actual_sentiment"].eq("").all()


def test_evaluation_returns_accuracy_macro_f1_and_confusion_matrix():
    labels = pd.DataFrame(
        {
            "sentiment_category": ["Positive", "Negative", "Neutral", "Positive"],
            "actual_sentiment": ["Positive", "Negative", "Neutral", "Negative"],
        }
    )

    summary, per_label, matrix = evaluate_sentiment_predictions(labels)

    assert summary["accuracy"] == 0.75
    assert 0 <= summary["macro_f1"] <= 1
    assert summary["model_coverage"] == 1
    assert set(per_label["label"]) == {"Positive", "Neutral", "Negative"}
    assert matrix.loc["Negative", "Positive"] == 1


def test_evaluation_rejects_missing_or_invalid_human_labels():
    labels = pd.DataFrame(
        {
            "sentiment_category": ["Positive"],
            "actual_sentiment": ["Mixed"],
        }
    )

    with pytest.raises(LabelValidationError):
        evaluate_sentiment_predictions(labels)


def test_human_labels_are_matched_to_current_predictions_by_comment_id():
    labels = pd.DataFrame(
        {"comment_id": ["a", "b"], "actual_sentiment": ["Positive", "Negative"]}
    )
    predictions = pd.DataFrame(
        {"comment_id": ["a", "b"], "sentiment_category": ["Positive", "Neutral"]}
    )

    matched = attach_model_predictions(labels, predictions)

    assert matched["sentiment_category"].tolist() == ["Positive", "Neutral"]


def test_evaluation_counts_uncertain_predictions_as_uncovered():
    labels = pd.DataFrame(
        {
            "sentiment_category": ["Uncertain", "Negative"],
            "actual_sentiment": ["Positive", "Negative"],
        }
    )

    summary, _, matrix = evaluate_sentiment_predictions(labels)

    assert summary["model_coverage"] == 0.5
    assert matrix.loc["Positive", "Uncertain"] == 1


def test_training_requires_enough_labeled_comments():
    labels = pd.DataFrame(
        {
            "cleaned_text": [f"comment {index}" for index in range(3)],
            "sentiment_category": ["Positive", "Neutral", "Negative"],
            "actual_sentiment": ["Positive", "Neutral", "Negative"],
        }
    )

    with pytest.raises(TrainingDataError):
        split_labeled_comments(labels)


def test_tweeteval_preparation_preserves_official_splits_and_label_mapping():
    dataset = {
        "train": {"text": ["good", "okay", "bad"], "label": [2, 1, 0]},
        "validation": {"text": ["nice", "fine", "awful"], "label": [2, 1, 0]},
        "test": {"text": ["love", "meh", "hate"], "label": [2, 1, 0]},
    }

    frame = prepare_tweeteval_frame(dataset)

    assert set(frame["dataset_split"]) == {"train", "validation", "test"}
    assert frame.loc[frame["cleaned_text"] == "good", "actual_sentiment"].item() == "Positive"


def test_training_uses_dataset_split_when_present():
    labels = pd.DataFrame(
        {
            "cleaned_text": [f"comment {index}" for index in range(90)],
            "actual_sentiment": ["Positive", "Neutral", "Negative"] * 30,
            "dataset_split": ["train"] * 60 + ["validation"] * 15 + ["test"] * 15,
        }
    )

    splits = split_labeled_comments(labels)

    assert len(splits.train) == 60
    assert len(splits.validation) == 15
    assert len(splits.test) == 15
