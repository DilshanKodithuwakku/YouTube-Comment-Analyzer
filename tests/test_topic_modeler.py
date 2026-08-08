import pandas as pd

from topic_modeler import model_topics


def test_model_topics_assigns_every_nonempty_comment_and_summarises():
    comments = pd.DataFrame(
        {
            "cleaned_text": [
                "python tutorial variables basics",
                "python tutorial functions basics",
                "camera lens focus quality",
                "camera lens price quality",
            ],
            "sentiment_score": [0.2, 0.1, -0.1, 0.0],
        }
    )

    assignments, summary = model_topics(comments, n_topics=2)

    assert assignments["topic_id"].notna().all()
    assert summary["comment_count"].sum() == len(comments)
    assert len(summary) == 2
