import pandas as pd
import re
import emoji
from textblob import TextBlob

# --- CONFIGURATION ---
INPUT_FILE = "youtube_comments.csv"
OUTPUT_FILE = "cleaned_comments.csv"

def clean_text(text):
    """
    Standard cleaning pipeline for social media text.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Convert Emojis to Text (e.g., 🔥 -> :fire:)
    # This is crucial because standard models ignore emojis, losing sentiment.
    text = emoji.demojize(text)
    
    # 2. Remove URLs (http://...)
    text = re.sub(r'http\S+', '', text)
    
    # 3. Remove User Mentions (@username)
    text = re.sub(r'@\w+', '', text)
    
    # 4. Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def get_sentiment(text):
    """
    Returns a polarity score between -1 (Negative) and 1 (Positive).
    """
    blob = TextBlob(text)
    return blob.sentiment.polarity

def process_data():
    print("--- Starting Data Processing ---")
    
    # 1. Load Data
    try:
        df = pd.read_csv(INPUT_FILE)
        print(f"Loaded {len(df)} raw comments.")
    except FileNotFoundError:
        print("Error: 'youtube_comments.csv' not found. Run data_collector.py first.")
        return

    # 2. Apply Cleaning
    # We use .apply() which is faster than a for-loop
    print("Cleaning text...")
    df['cleaned_text'] = df['text'].apply(clean_text)

    # 3. Filter Garbage
    # Remove rows where comments are empty or too short (less than 2 chars)
    initial_count = len(df)
    df = df[df['cleaned_text'].str.len() > 2]
    print(f"Removed {initial_count - len(df)} empty/short comments.")

    # 4. Add Sentiment Score
    print("Calculating sentiment...")
    df['sentiment_score'] = df['cleaned_text'].apply(get_sentiment)

    # 5. Classify Sentiment (Optional helper column)
    def categorize(score):
        if score > 0.1: return 'Positive'
        if score < -0.1: return 'Negative'
        return 'Neutral'
    
    df['sentiment_category'] = df['sentiment_score'].apply(categorize)

    # 6. Save
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Success! Saved processed data to '{OUTPUT_FILE}'")
    print(df[['text', 'sentiment_category', 'sentiment_score']].head())

if __name__ == "__main__":
    process_data()