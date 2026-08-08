"""Local Transformer inference for the project's trained sentiment model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from sentiment_evaluation import SENTIMENT_LABELS


class ModelUnavailableError(RuntimeError):
    """Raised when a local production sentiment model cannot be loaded."""


def is_trained_model(model_path: Path) -> bool:
    """Return whether a Hugging Face model config exists at ``model_path``."""
    return (model_path / "config.json").is_file()


@dataclass
class TransformerSentimentPredictor:
    """Batch prediction wrapper with a confidence threshold for uncertain text."""

    model: object
    tokenizer: object
    batch_size: int = 32

    @classmethod
    def load(cls, model_path: Path, batch_size: int = 32) -> "TransformerSentimentPredictor":
        """Load a fine-tuned local model without contacting a remote inference API."""
        if not is_trained_model(model_path):
            raise ModelUnavailableError(
                f"No trained model was found at {model_path}. Train one before selecting it."
            )
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as error:
            raise ModelUnavailableError(
                "Transformer support is not installed. Run pip install -r requirements.txt."
            ) from error

        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
        model.eval()
        model_labels = {str(label) for label in model.config.id2label.values()}
        missing_labels = set(SENTIMENT_LABELS).difference(model_labels)
        if missing_labels:
            raise ModelUnavailableError(
                "The trained model must use these labels: " + ", ".join(SENTIMENT_LABELS)
            )
        return cls(model=model, tokenizer=tokenizer, batch_size=batch_size)

    def predict(self, texts: Iterable[str], confidence_threshold: float = 0.60) -> pd.DataFrame:
        """Return sentiment category, signed score, and confidence for each input text."""
        if not 0 < confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be greater than 0 and no more than 1.")
        try:
            import torch
        except ImportError as error:
            raise ModelUnavailableError(
                "PyTorch is required for Transformer sentiment inference."
            ) from error

        values = [str(text) for text in texts]
        label_lookup = {int(index): label for index, label in self.model.config.id2label.items()}
        rows: list[dict[str, object]] = []
        for start in range(0, len(values), self.batch_size):
            batch = values[start : start + self.batch_size]
            encoded = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                probabilities = torch.softmax(self.model(**encoded).logits, dim=1).cpu()
            for probability in probabilities:
                confidence, label_id = torch.max(probability, dim=0)
                category = label_lookup[int(label_id)]
                confidence_value = float(confidence)
                negative = float(probability[[key for key, value in label_lookup.items() if value == "Negative"][0]])
                positive = float(probability[[key for key, value in label_lookup.items() if value == "Positive"][0]])
                rows.append(
                    {
                        "sentiment_category": category if confidence_value >= confidence_threshold else "Uncertain",
                        "sentiment_score": positive - negative,
                        "sentiment_confidence": confidence_value,
                        "sentiment_model": "fine_tuned_transformer",
                    }
                )
        return pd.DataFrame(rows)
