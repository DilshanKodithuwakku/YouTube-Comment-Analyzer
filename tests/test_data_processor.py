import pandas as pd

from data_processor import clean_text, process_dataframe


def test_clean_text_removes_urls_mentions_and_keeps_emoji_meaning():
    assert clean_text("Great 🔥 @channel https://example.com\n") == "Great :fire:"


def test_process_dataframe_deduplicates_ids_and_assigns_sentiment():
    raw = pd.DataFrame(
        {
            "comment_id": ["a", "a", "b", "c"],
            "text": ["I love this tutorial", "I love this tutorial", "I hate this", "x"],
            "likes": ["2", "2", "not-a-number", "0"],
        }
    )

    processed = process_dataframe(raw)

    assert len(processed) == 2
    assert set(processed["sentiment_category"]) == {"Positive", "Negative"}
    assert processed["likes"].tolist() == [2, 0]


def test_process_dataframe_accepts_a_production_sentiment_predictor():
    raw = pd.DataFrame({"text": ["I love it", "This is bad"]})

    def predictor(texts):
        assert texts == ["I love it", "This is bad"]
        return pd.DataFrame(
            {
                "sentiment_category": ["Positive", "Uncertain"],
                "sentiment_score": [0.91, -0.02],
                "sentiment_confidence": [0.97, 0.54],
                "sentiment_model": ["fine_tuned_transformer", "fine_tuned_transformer"],
            }
        )

    processed = process_dataframe(raw, sentiment_predictor=predictor)

    assert processed["sentiment_category"].tolist() == ["Positive", "Uncertain"]
    assert processed["sentiment_confidence"].tolist() == [0.97, 0.54]
