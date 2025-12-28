# 📺 YouTube Comment Analyzer
### *Turning Social Noise into Actionable Insights*

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white) ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white) ![NLP](https://img.shields.io/badge/AI-NLP-green?style=for-the-badge)

## 📖 Overview
A full-stack Data Science application that extracts thousands of comments from YouTube videos to uncover viewer sentiment and trending topics.

Unlike standard analytics that only show "likes" or "views," this tool uses **Natural Language Processing (NLP)** to understand the *quality* of the engagement. It helps creators and brands understand the "vibe" of their audience and cluster discussions into meaningful themes.

## 📊 Key Features
* **Data Extraction:** Automated scraping of comments using the **YouTube Data API v3**.
* **Sentiment Analysis:** Categorizes comments as *Positive*, *Negative*, or *Neutral* using Polarity scores (TextBlob).
* **Topic Modeling:** Uses **NMF (Non-Negative Matrix Factorization)** to group comments into hidden themes (e.g., "Audio Quality", "Pricing", "Tutorial Help").
* **Interactive Dashboard:** Built with **Streamlit** for real-time filtering and visualization.

## 🛠️ Tech Stack
* **Language:** Python
* **Data Collection:** `google-api-python-client`
* **Processing:** `pandas`, `numpy`, `emoji`
* **NLP:** `TextBlob`, `NLTK`, `Scikit-Learn` (TF-IDF & NMF)
* **Visualization:** `Streamlit`, `Matplotlib`, `WordCloud`

## 🚀 How to Run Locally

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/youtube-comment-analyzer.git](https://github.com/YOUR_USERNAME/youtube-comment-analyzer.git)
cd youtube-comment-analyzer