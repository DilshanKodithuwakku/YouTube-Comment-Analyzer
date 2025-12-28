import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="YouTube Comment Analyzer", page_icon="📺", layout="wide")

# --- TITLE & SIDEBAR ---
st.title("📺 YouTube Comment Analyzer: Turning Social Noise into Actionable Insights.")
st.markdown("Analyze the *real* sentiment and topics of any YouTube video.")

st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload processed CSV", type=['csv'])

# --- MAIN LOGIC ---
if uploaded_file is not None:
    # Load Data
    df = pd.read_csv(uploaded_file)
    
    # 1. TOP METRICS
    total_comments = len(df)
    positive_comments = len(df[df['sentiment_category'] == 'Positive'])
    negative_comments = len(df[df['sentiment_category'] == 'Negative'])
    avg_sentiment = df['sentiment_score'].mean()

    # Display Metrics in Columns
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Comments", total_comments)
    col2.metric("Positive Vibes", f"{positive_comments} ({round(positive_comments/total_comments*100)}%)")
    col3.metric("Negative Vibes", f"{negative_comments} ({round(negative_comments/total_comments*100)}%)")
    col4.metric("Avg Sentiment Score", f"{avg_sentiment:.2f}")

    st.divider()

    # 2. VISUALIZATIONS
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Sentiment Distribution")
        # Simple Bar Chart
        sentiment_counts = df['sentiment_category'].value_counts()
        st.bar_chart(sentiment_counts)

    with col_right:
        st.subheader("Sample Comments")
        filter_option = st.selectbox("Filter by Vibe:", ["Negative", "Positive", "Neutral"])
        
        # Show top 5 comments for selected vibe
        filtered_df = df[df['sentiment_category'] == filter_option]
        for i, row in filtered_df.head(5).iterrows():
            st.text_area(f"{row['author']} (Score: {row['sentiment_score']:.2f})", row['cleaned_text'], height=70)

    # 3. RAW DATA (Optional)
    with st.expander("View Raw Data"):
        st.dataframe(df)

else:
    st.info("👈 Please upload the 'cleaned_comments.csv' file generated in Step 2 to begin.")