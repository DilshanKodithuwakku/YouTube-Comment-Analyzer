"""Streamlit dashboard for explored, processed YouTube comments."""

from __future__ import annotations

from io import BytesIO
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from data_collector import CommentCollectionError, ConfigurationError, get_video_comments
from data_processor import DataValidationError, SENTIMENT_ORDER, process_dataframe
from sentiment_evaluation import (
    LabelValidationError,
    attach_model_predictions,
    create_labeling_template,
    evaluate_sentiment_predictions,
)
from sentiment_model import ModelUnavailableError, TransformerSentimentPredictor, is_trained_model
from topic_modeler import TopicModelError, model_topics

st.set_page_config(
    page_title="YouTube Comment Analyzer",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="expanded",
)

REQUIRED_COLUMNS = {"cleaned_text", "sentiment_category", "sentiment_score"}
DEFAULT_MODEL_DIR = Path("models/sentiment")


@st.cache_data(show_spinner=False)
def load_comments(contents: bytes) -> pd.DataFrame:
    """Load uploaded analysis data once per distinct CSV."""
    return pd.read_csv(BytesIO(contents))


@st.cache_data(show_spinner="Discovering recurring topics…")
def discover_topics(data: pd.DataFrame, topic_count: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cache deterministic NMF topic analysis for the current dashboard filters."""
    return model_topics(data, n_topics=topic_count)


@st.cache_resource(show_spinner="Loading the fine-tuned sentiment model...")
def load_sentiment_predictor(model_dir: str) -> TransformerSentimentPredictor:
    """Load the local trained model once per Streamlit server process."""
    return TransformerSentimentPredictor.load(Path(model_dir))


def validate_data(data: pd.DataFrame) -> list[str]:
    """Return dashboard-compatible CSV schema errors, if any."""
    missing = REQUIRED_COLUMNS.difference(data.columns)
    errors: list[str] = []
    if missing:
        errors.append(f"Missing required column(s): {', '.join(sorted(missing))}.")
    if not data.empty and "sentiment_score" in data.columns:
        converted = pd.to_numeric(data["sentiment_score"], errors="coerce")
        if converted.isna().all():
            errors.append("sentiment_score must contain numeric values.")
    return errors


def sentiment_percent(data: pd.DataFrame, category: str) -> float:
    """Return the percentage of a sentiment category without zero-division risk."""
    if data.empty:
        return 0.0
    return (data["sentiment_category"] == category).mean() * 100


def render_overview(data: pd.DataFrame) -> None:
    """Render headline metrics and the sentiment distribution chart."""
    likes = pd.to_numeric(data["likes"], errors="coerce").fillna(0) if "likes" in data else pd.Series(dtype=int)
    metrics = st.columns(5)
    metrics[0].metric("Comments", f"{len(data):,}")
    metrics[1].metric("Positive", f"{sentiment_percent(data, 'Positive'):.0f}%")
    metrics[2].metric("Negative", f"{sentiment_percent(data, 'Negative'):.0f}%")
    metrics[3].metric("Uncertain", f"{sentiment_percent(data, 'Uncertain'):.0f}%")
    metrics[4].metric("Average sentiment", f"{data['sentiment_score'].mean():.2f}")
    st.caption(f"Comment likes in the current selection: {int(likes.sum()):,}")
    if "sentiment_model" in data.columns:
        st.caption("Sentiment engine: " + ", ".join(sorted(data["sentiment_model"].dropna().unique())))

    st.subheader("Sentiment distribution")
    counts = data["sentiment_category"].value_counts().reindex(SENTIMENT_ORDER, fill_value=0)
    st.bar_chart(counts, x_label="Sentiment", y_label="Comments")


def render_comments(data: pd.DataFrame) -> None:
    """Render sortable, consistently scoped comment details."""
    st.subheader("Comments")
    display_columns = [
        column
        for column in [
            "author",
            "cleaned_text",
            "sentiment_category",
            "sentiment_confidence",
            "sentiment_score",
            "likes",
            "published_at",
        ]
        if column in data.columns
    ]
    sortable = data.copy()
    if "likes" in sortable.columns:
        sortable["likes"] = pd.to_numeric(sortable["likes"], errors="coerce").fillna(0)
        sortable = sortable.sort_values("likes", ascending=False)
    visible_comments = sortable.head(1_000)
    if len(sortable) > len(visible_comments):
        st.caption(f"Showing the 1,000 most-liked comments out of {len(sortable):,}.")
    st.dataframe(visible_comments[display_columns], hide_index=True)


def render_topics(data: pd.DataFrame) -> None:
    """Run and display local topic analysis for the currently filtered comments."""
    st.subheader("Discussion topics")
    st.caption("Topics use deterministic TF-IDF + NMF locally; no comments are sent to another service.")
    if not st.session_state.get("topics_requested", False):
        if st.button("Discover discussion topics", key="discover_topics_button"):
            st.session_state["topics_requested"] = True
        else:
            st.info("Select Discover discussion topics to run the local topic model for the current filters.")
            return
    max_topics = min(10, max(2, len(data)))
    topic_count = st.slider("Number of topics", min_value=2, max_value=max_topics, value=min(5, max_topics))
    try:
        assignments, summary = discover_topics(data, topic_count)
    except TopicModelError as error:
        st.info(f"Topics are unavailable for this selection: {error}")
        return

    st.dataframe(summary, hide_index=True)
    st.bar_chart(summary.set_index("topic_label")["comment_count"], y_label="Comments")
    with st.expander("View topic assignments"):
        columns = [
            column
            for column in ["cleaned_text", "topic_id", "topic_label", "topic_confidence", "sentiment_category"]
            if column in assignments.columns
        ]
        visible_assignments = assignments.head(1_000)
        if len(assignments) > len(visible_assignments):
            st.caption(f"Showing the first 1,000 assignments out of {len(assignments):,}.")
        st.dataframe(visible_assignments[columns], hide_index=True)


def analyze_video(video_url: str, sentiment_predictor=None) -> pd.DataFrame:
    """Collect and process every public comment for the supplied YouTube video."""
    load_dotenv()
    progress_message = st.empty()

    def show_collection_progress(comment_count: int) -> None:
        progress_message.info(f"Downloaded {comment_count:,} public comments so far...")

    raw_comments = get_video_comments(
        video_url,
        os.getenv("YOUTUBE_API_KEY", ""),
        max_results=None,
        include_replies=True,
        progress_callback=show_collection_progress,
    )
    progress_message.success(f"Downloaded {len(raw_comments):,} public comments.")
    with st.spinner(f"Cleaning and scoring {len(raw_comments):,} comments..."):
        return process_dataframe(raw_comments, sentiment_predictor=sentiment_predictor)


def render_model_quality(comments: pd.DataFrame) -> None:
    """Give operators a repeatable human-labelling and evaluation workflow."""
    st.subheader("Measure sentiment quality")
    st.write(
        "Label a representative sample independently, then evaluate the model against those human labels. "
        "Macro F1 is the main score because it gives each sentiment class equal weight."
    )
    sample_limit = min(500, len(comments))
    if sample_limit < 25:
        sample_size = sample_limit
        st.caption(f"Only {sample_size:,} comments are available, so all of them are included for labelling.")
    else:
        sample_size = st.slider(
            "Comments to label",
            min_value=25,
            max_value=sample_limit,
            value=min(200, sample_limit),
            key="label_sample_size",
        )
    template = create_labeling_template(comments, sample_size)
    st.download_button(
        "Download labelling template",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="sentiment_labelling_template.csv",
        mime="text/csv",
        key="download_label_template",
    )
    st.caption("Fill actual_sentiment with Positive, Neutral, or Negative. Predictions are intentionally hidden from reviewers.")

    uploaded_labels = st.file_uploader(
        "Upload independently labelled comments", type=["csv"], key="labelled_comments_upload"
    )
    if uploaded_labels is None:
        return
    try:
        labeled_comments = attach_model_predictions(pd.read_csv(uploaded_labels), comments)
        summary, per_label, matrix = evaluate_sentiment_predictions(labeled_comments)
    except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError, LabelValidationError) as error:
        st.error(f"The labelled CSV could not be evaluated: {error}")
        return

    metrics = st.columns(4)
    metrics[0].metric("Labelled comments", f"{summary['sample_size']:,.0f}")
    metrics[1].metric("Accuracy", f"{summary['accuracy']:.1%}")
    metrics[2].metric("Macro F1", f"{summary['macro_f1']:.1%}")
    metrics[3].metric("Model coverage", f"{summary['model_coverage']:.1%}")
    st.caption("Per-label precision and recall identify which sentiment classes need more training examples.")
    st.dataframe(per_label, hide_index=True)
    st.caption("Confusion matrix: rows are human labels; columns are predictions.")
    st.dataframe(matrix)
    st.code(
        "python train_sentiment_model.py --input sentiment_labelling_template.csv "
        "--output-dir models/sentiment",
        language="powershell",
    )


def main() -> None:
    st.session_state.setdefault("analyzed_comments", None)
    st.session_state.setdefault("analyzed_video_url", "")
    st.session_state.setdefault("topics_requested", False)

    st.title("YouTube Comment Analyzer")
    st.write("Paste a public YouTube video URL to analyze its available comments, sentiment, and discussion themes.")

    with st.sidebar:
        st.header("Data source")
        source = st.radio("Analyze from", ["YouTube URL", "Processed CSV"])

        if source == "YouTube URL":
            sentiment_engine = "TextBlob baseline"
            confidence_threshold = 0.60
            if is_trained_model(DEFAULT_MODEL_DIR):
                sentiment_engine = st.selectbox(
                    "Sentiment engine",
                    ["Fine-tuned Transformer", "TextBlob baseline"],
                )
                if sentiment_engine == "Fine-tuned Transformer":
                    confidence_threshold = st.slider(
                        "Minimum model confidence",
                        min_value=0.50,
                        max_value=0.95,
                        value=0.60,
                        step=0.05,
                    )
            else:
                st.caption("Using the TextBlob baseline until a trained model exists in models/sentiment.")
            with st.form("video_analysis_form", border=False):
                video_url = st.text_input(
                    "YouTube video URL",
                    placeholder="https://www.youtube.com/watch?v=VIDEO_ID",
                )
                analyze_clicked = st.form_submit_button("Analyze all public comments", type="primary")
            st.caption("This includes top-level comments and replies. Large videos can take several minutes.")
        else:
            uploaded_file = st.file_uploader(
                "Upload processed comments", type=["csv"], help="Create this file with data_processor.py."
            )
            st.caption("The file must include cleaned_text, sentiment_score, and sentiment_category.")

    comments: pd.DataFrame | None = None
    if source == "YouTube URL":
        if analyze_clicked:
            if not video_url.strip():
                st.warning("Paste a YouTube video URL first.")
                return
            try:
                predictor = None
                if sentiment_engine == "Fine-tuned Transformer":
                    trained_model = load_sentiment_predictor(str(DEFAULT_MODEL_DIR))
                    predictor = lambda texts: trained_model.predict(texts, confidence_threshold)
                comments = analyze_video(video_url, predictor)
            except (
                ConfigurationError,
                CommentCollectionError,
                DataValidationError,
                ModelUnavailableError,
                ValueError,
            ) as error:
                st.error(str(error))
                return
            st.session_state["analyzed_comments"] = comments
            st.session_state["analyzed_video_url"] = video_url
            st.session_state["topics_requested"] = False
        else:
            comments = st.session_state.get("analyzed_comments")
            if comments is None:
                st.info("Paste a public YouTube video URL, then choose **Analyze all public comments**.")
                return
            st.caption(f"Showing the last analysis: {st.session_state.get('analyzed_video_url', '')}")
    else:
        if uploaded_file is None:
            st.info("Upload a processed CSV to start exploring audience sentiment and themes.")
            return
        try:
            comments = load_comments(uploaded_file.getvalue())
        except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
            st.error(f"The uploaded file could not be read as a CSV: {error}")
            return

    errors = validate_data(comments)
    if errors:
        st.error(" ".join(errors))
        return
    if comments.empty:
        st.warning("No public comments were available for this video.")
        return

    comments = comments.copy()
    comments["sentiment_score"] = pd.to_numeric(comments["sentiment_score"], errors="coerce")
    comments = comments.dropna(subset=["sentiment_score", "cleaned_text"])
    if comments.empty:
        st.warning("No rows contain both text and a numeric sentiment score.")
        return

    available_sentiments = [
        category for category in SENTIMENT_ORDER if category in comments["sentiment_category"].unique()
    ]
    with st.sidebar:
        st.header("Filters")
        selected_sentiments = st.multiselect(
            "Sentiment", options=available_sentiments, default=available_sentiments
        )
        minimum_likes = 0
        if "likes" in comments.columns:
            comments["likes"] = pd.to_numeric(comments["likes"], errors="coerce").fillna(0).astype(int)
            minimum_likes = st.number_input("Minimum likes", min_value=0, value=0, step=1)

    filtered = comments.loc[comments["sentiment_category"].isin(selected_sentiments)].copy()
    if "likes" in filtered.columns:
        filtered = filtered.loc[filtered["likes"] >= minimum_likes]
    if filtered.empty:
        st.warning("No comments match the selected filters.")
        return

    overview_tab, comments_tab, topics_tab, quality_tab, data_tab = st.tabs(
        ["Overview", "Comments", "Topics", "Model quality", "Data"]
    )
    with overview_tab:
        render_overview(filtered)
    with comments_tab:
        render_comments(filtered)
    with topics_tab:
        render_topics(filtered)
    with quality_tab:
        render_model_quality(comments)
    with data_tab:
        visible_data = filtered.head(1_000)
        if len(filtered) > len(visible_data):
            st.caption(f"Showing the first 1,000 rows out of {len(filtered):,}; the download contains all rows.")
        st.dataframe(visible_data, hide_index=True)
        st.download_button(
            "Download filtered comments",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="filtered_youtube_comments.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
