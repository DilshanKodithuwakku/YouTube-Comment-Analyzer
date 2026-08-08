# YouTube Comment Analyzer

Analyze the sentiment and recurring discussion themes in the public comments of a YouTube video. The project uses the official YouTube Data API v3, processes comment data locally, and presents the result in a Streamlit dashboard.

## What it does

- Fetches all available public comments by default, including top-level comments and replies, along with author, likes, dates, and stable YouTube comment IDs.
- Normalizes social text without discarding emoji meaning, removes URLs and mentions, and produces baseline TextBlob polarity labels when no trained model is available.
- Finds repeatable local topics with TF-IDF and Non-negative Matrix Factorization (NMF). This does not download an embedding model or send comment content to another service.
- Provides a dashboard with filters, headline metrics, sentiment distribution, topic assignments, CSV export, and a human-labelling quality workflow.
- Supports a fine-tuned local Transformer model for production sentiment predictions with confidence-based `Uncertain` results.

## Project structure

```text
data_collector.py   Fetch raw comments from the YouTube API
data_processor.py   Clean comments and assign sentiment scores
sentiment_model.py  Run the locally trained Transformer model
sentiment_evaluation.py  Create label samples and calculate accuracy / macro F1
train_sentiment_model.py Fine-tune and evaluate the production sentiment model
prepare_tweeteval_dataset.py Download the public TweetEval sentiment training dataset
topic_modeler.py    Create topic assignments and topic summary CSVs
app.py              Paste a URL and explore the resulting analysis in Streamlit
tests/              Unit tests for core data logic
```

Generated data is kept outside source files:

```text
data/raw/youtube_comments.csv
data/processed/cleaned_comments.csv
data/processed/topic_comments.csv
data/processed/topic_summary.csv
```

## Quick start

This project requires Python 3.10 or newer.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Create a YouTube Data API v3 key in Google Cloud with the YouTube Data API enabled, then replace the placeholder in `.env`. Do not commit that file.

Start the dashboard, paste a public YouTube URL, and select **Analyze all public comments**:

```powershell
streamlit run app.py
```

The dashboard keeps the downloaded analysis in the current browser session and lets you download the filtered result as a CSV. Large videos can take several minutes to fetch and process.

## Improve sentiment quality for production

The initial dashboard uses a TextBlob baseline. It is useful for an early demo but is not a validated production model. To train a local production model:

1. In the dashboard, open **Model quality** and download a labelling template containing a representative random sample of comments.
2. Have people independently fill in the `actual_sentiment` column using only `Positive`, `Neutral`, or `Negative`. The exported template hides the prediction to reduce reviewer bias.
3. Save the completed CSV, upload it to **Model quality**, and inspect accuracy, macro F1, model coverage, per-label precision/recall, and the confusion matrix. Coverage reports the percentage of comments for which the model made a confident prediction.
4. Train the local multilingual model. It reserves 15% of your labelled data as a final test set and saves that untouched test result with the model:

```powershell
python train_sentiment_model.py --input "data/labels/labeled_comments.csv" --output-dir models/sentiment
```

5. Restart the dashboard. It detects `models/sentiment` and lets you choose **Fine-tuned Transformer**. Low-confidence predictions are shown as `Uncertain` instead of pretending to be correct.

The default base model is XLM-RoBERTa, selected for multilingual comment analysis. Training downloads the base model on first use and needs substantially more memory and time than the baseline. Start with several hundred human labels per class; grow the dataset with comments where the model is uncertain or commonly wrong.

For a quicker CPU-only first model using English TweetEval data, use the smaller DistilBERT profile. It trains for one epoch with a larger batch size and shorter input limit, so it is not a substitute for the multilingual production model:

```powershell
python train_sentiment_model.py --input data/labels/tweeteval_sentiment.csv --output-dir models/sentiment --cpu-fast
```

Training saves a resumable checkpoint every 500 steps. Re-running the same command after an interruption resumes from the latest checkpoint automatically.

### Train an initial model from public data

If you cannot label comments yet, train an initial English social-media model from the public TweetEval sentiment dataset. It contains the official three-class Negative, Neutral, and Positive splits, which this project preserves during training:

```powershell
python prepare_tweeteval_dataset.py
python train_sentiment_model.py --input data/labels/tweeteval_sentiment.csv --output-dir models/sentiment
```

The first command downloads the dataset; the second downloads the Transformer base model and trains it locally. The resulting `models/sentiment/test_metrics.json` measures performance on TweetEval's public test set only. It is not a verified accuracy score for your own YouTube videos.

For batch processing with the trained model:

```powershell
python data_processor.py --model-dir models/sentiment --confidence-threshold 0.60
```

For command-line use, the collector also accepts a bare 11-character video ID and common YouTube share, Shorts, embed, and live URLs. It fetches every available public top-level comment and reply when `--max-results` is omitted:

```powershell
python data_collector.py --url "https://www.youtube.com/watch?v=VIDEO_ID"
python data_processor.py
python topic_modeler.py --topics 5
```

Use `--max-results 500` to set a limit, or `--top-level-only` to exclude replies. The collector loads `.env` when `YOUTUBE_API_KEY` is not already defined; an explicit `--api-key` flag takes precedence.

## Inputs and outputs

`data_processor.py` expects a raw CSV with a `text` column. It creates these fields:

| Field | Meaning |
| --- | --- |
| `cleaned_text` | Normalized comment content used for analysis |
| `sentiment_score` | Signed sentiment score from -1 (negative) to 1 (positive); TextBlob polarity for the baseline or Positive-minus-Negative probability for the trained model |
| `sentiment_category` | Positive, Neutral, Negative, or `Uncertain` for a low-confidence trained-model prediction |
| `sentiment_confidence` | Highest trained-model class probability; empty for the TextBlob baseline |
| `sentiment_model` | `textblob_baseline` or `fine_tuned_transformer` |

The dashboard requires those three fields and additionally uses `likes`, `author`, and `published_at` when present.

## Quality and operational notes

- Sentiment is lexicon-based and works best for English; treat it as a directional signal, not a ground-truth measure of intent, sarcasm, or abuse.
- Production quality is measured only against independently human-labelled data. Report macro F1, per-label precision/recall, accuracy, the confidence threshold, model version, test-set size, and the languages represented in the test data.
- API quotas, comment permissions, deleted videos, and private videos can prevent collection. The collector reports these as errors rather than silently returning an empty report.
- Public comments that YouTube makes available through its API are included, including replies. Deleted, held-for-review, disabled, private, or otherwise unavailable comments cannot be collected.
- Run the automated checks with `pytest`.

## License

Add a license before distributing or accepting outside contributions.
