import os
import googleapiclient.discovery
import pandas as pd
from urllib.parse import urlparse, parse_qs

# --- CONFIGURATION ---
API_KEY = "PASTE_YOUR_YOUTUBE_API_KEY_HERE"  # Replace with your YouTube Data API v3 key

def extract_video_id(url):
    """
    Extracts the 'v' parameter from a YouTube URL.
    Example: https://www.youtube.com/watch?v=kqtD5dpn9C8 -> kqtD5dpn9C8
    """
    query = urlparse(url).query
    params = parse_qs(query)
    return params["v"][0] if "v" in params else url

def get_video_comments(video_url, api_key, max_results=200):
    # Extract ID
    video_id = extract_video_id(video_url)
    
    # Disable OAuthlib's HTTPS verification when running locally.
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    api_service_name = "youtube"
    api_version = "v3"

    try:
        youtube = googleapiclient.discovery.build(
            api_service_name, api_version, developerKey=api_key)

        comments = []
        
        # Initial request
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            textFormat="plainText"
        )

        print(f"--- Fetching comments for ID: {video_id} ---")

        # Loop through pages
        while request and len(comments) < max_results:
            response = request.execute()

            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']
                comments.append({
                    'author': comment['authorDisplayName'],
                    'text': comment['textDisplay'],
                    'likes': comment['likeCount'],
                    'date': comment['publishedAt']
                })

            # Check for next page
            if 'nextPageToken' in response:
                request = youtube.commentThreads().list_next(request, response)
            else:
                break
                
        print(f"Done! Fetched {len(comments)} comments.")
        return pd.DataFrame(comments)

    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Test with a video URL
    TARGET_URL = "https://www.youtube.com/watch?v=kqtD5dpn9C8" # Python Tutorial
    
    df = get_video_comments(TARGET_URL, API_KEY)
    
    if not df.empty:
        # Save to CSV in the same folder
        df.to_csv("youtube_comments.csv", index=False)
        print("Success! Data saved to 'youtube_comments.csv'")
    else:
        print("No data found or API Key is invalid.")