import pandas as pd
from bertopic import BERTopic

# --- CONFIGURATION ---
INPUT_FILE = "cleaned_comments.csv"
HTML_OUTPUT = "topics_visualization.html"

def run_topic_modeling():
    print("--- Starting Topic Modeling ---")
    
    # 1. Load Data
    try:
        df = pd.read_csv(INPUT_FILE)
        # We need the text as a list of strings, not a DataFrame column
        docs = df['cleaned_text'].tolist()
        print(f"Loaded {len(docs)} comments.")
    except FileNotFoundError:
        print("Error: 'cleaned_comments.csv' not found.")
        return

    # 2. Initialize Model
    # verbose=True lets you see the progress bar
    # min_topic_size=5 means a topic needs at least 5 comments to count
    print("Initializing BERTopic (this downloads the AI model on first run)...")
    topic_model = BERTopic(language="english", min_topic_size=5, verbose=True)

    # 3. Fit the Model (The "Learning" Phase)
    print("Fitting the model... (This might take a minute)")
    topics, probs = topic_model.fit_transform(docs)

    # 4. Display Results
    print("\n--- Top Topics Found ---")
    # Get the overview of topics
    freq = topic_model.get_topic_info()
    print(freq.head(10))  # Print top 10 topics to console

    # 5. Save Visualization
    # This creates an interactive bubble chart
    print(f"\nSaving interactive map to {HTML_OUTPUT}...")
    fig = topic_model.visualize_topics()
    fig.write_html(HTML_OUTPUT)
    
    print("Done! Open 'topics_visualization.html' in your browser to see the clusters.")

if __name__ == "__main__":
    run_topic_modeling()