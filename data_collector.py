"""Collect top-level YouTube comments with the YouTube Data API v3.

The module is deliberately usable both from the command line and from another
Python application. API credentials are read from ``YOUTUBE_API_KEY`` unless
they are passed explicitly; credentials are never stored in source control.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

import googleapiclient.discovery
from googleapiclient.errors import HttpError
import pandas as pd
from dotenv import load_dotenv

LOGGER = logging.getLogger(__name__)
VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")
COMMENT_COLUMNS = [
    "comment_id",
    "parent_comment_id",
    "comment_type",
    "video_id",
    "author",
    "text",
    "likes",
    "published_at",
    "updated_at",
]
ProgressCallback = Callable[[int], None]


class ConfigurationError(ValueError):
    """Raised when required user configuration is missing or malformed."""


class CommentCollectionError(RuntimeError):
    """Raised when YouTube rejects or cannot complete a comment request."""


def extract_video_id(video_url_or_id: str) -> str:
    """Return a valid ID from a YouTube watch, short, embed, live, or share URL.

    A bare 11-character video ID is also accepted. Invalid inputs raise a clear
    error instead of sending a malformed request to the YouTube API.
    """
    value = str(video_url_or_id).strip()
    if VIDEO_ID_PATTERN.fullmatch(value):
        return value

    parsed = urlparse(value)
    if not parsed.scheme:
        parsed = urlparse(f"https://{value}")

    host = parsed.netloc.lower().removeprefix("www.")
    path_parts = [part for part in parsed.path.split("/") if part]
    candidate = ""

    if host in {"youtube.com", "m.youtube.com", "music.youtube.com"}:
        candidate = parse_qs(parsed.query).get("v", [""])[0]
        if not candidate and len(path_parts) >= 2 and path_parts[0] in {"shorts", "embed", "live"}:
            candidate = path_parts[1]
    elif host == "youtu.be" and path_parts:
        candidate = path_parts[0]

    if VIDEO_ID_PATTERN.fullmatch(candidate):
        return candidate
    raise ConfigurationError(
        "Provide a valid YouTube video URL or an 11-character video ID."
    )


def _create_youtube_client(api_key: str) -> Any:
    if not api_key or api_key == "PASTE_YOUR_YOUTUBE_API_KEY_HERE":
        raise ConfigurationError(
            "A YouTube API key is required. Set YOUTUBE_API_KEY or pass --api-key."
        )
    return googleapiclient.discovery.build(
        "youtube", "v3", developerKey=api_key, cache_discovery=False
    )


def _comment_row(comment: dict[str, Any], video_id: str, comment_type: str) -> dict[str, Any] | None:
    """Convert one YouTube comment resource into the project's stable schema."""
    snippet = comment.get("snippet", {})
    comment_id = comment.get("id", "")
    if not snippet or not comment_id:
        return None
    return {
        "comment_id": comment_id,
        "parent_comment_id": snippet.get("parentId", ""),
        "comment_type": comment_type,
        "video_id": video_id,
        "author": snippet.get("authorDisplayName", "Unknown"),
        "text": snippet.get("textDisplay", ""),
        "likes": snippet.get("likeCount", 0),
        "published_at": snippet.get("publishedAt", ""),
        "updated_at": snippet.get("updatedAt", ""),
    }


def get_video_comments(
    video_url_or_id: str,
    api_key: str,
    max_results: int | None = None,
    include_replies: bool = True,
    progress_callback: ProgressCallback | None = None,
    youtube_client: Any | None = None,
) -> pd.DataFrame:
    """Fetch all public comments for one video, or stop at ``max_results``.

    Top-level comments are fetched through ``commentThreads.list``. When
    ``include_replies`` is true, every reply is fetched as well, including reply
    pages that are not embedded in a thread response. ``max_results=None``
    means no client-side limit. ``progress_callback`` receives the running
    number of unique comments after each fetched thread page.

    The returned frame always uses ``COMMENT_COLUMNS``, including when comments
    are disabled or a video has no comments. API failures are raised with a
    helpful message so callers do not mistake an error for an empty audience.
    """
    if max_results is not None and max_results < 1:
        raise ConfigurationError("max_results must be at least 1.")

    video_id = extract_video_id(video_url_or_id)
    youtube = youtube_client or _create_youtube_client(api_key)
    comments: dict[str, dict[str, Any]] = {}
    page_token: str | None = None

    def add_comment(comment: dict[str, Any], comment_type: str) -> bool:
        """Add a unique comment, returning whether the collection may continue."""
        row = _comment_row(comment, video_id, comment_type)
        if row and row["comment_id"] not in comments:
            comments[row["comment_id"]] = row
        return max_results is None or len(comments) < max_results

    def fetch_replies(parent_id: str) -> bool:
        """Retrieve every page of replies for one top-level comment."""
        reply_page_token: str | None = None
        while True:
            response = (
                youtube.comments()
                .list(
                    part="snippet",
                    parentId=parent_id,
                    maxResults=100,
                    pageToken=reply_page_token,
                    textFormat="plainText",
                )
                .execute()
            )
            for reply in response.get("items", []):
                if not add_comment(reply, "reply"):
                    return False
            reply_page_token = response.get("nextPageToken")
            if not reply_page_token:
                return True

    limit_description = f"up to {max_results:,}" if max_results is not None else "all available"
    LOGGER.info("Fetching %s comments for video %s", limit_description, video_id)
    try:
        while max_results is None or len(comments) < max_results:
            page_size = 100 if max_results is None else min(100, max_results - len(comments))
            response = (
                youtube.commentThreads()
                .list(
                    part="snippet,replies",
                    videoId=video_id,
                    maxResults=page_size,
                    pageToken=page_token,
                    textFormat="plainText",
                    order="time",
                )
                .execute()
            )

            for item in response.get("items", []):
                top_level_comment = item.get("snippet", {}).get("topLevelComment", {})
                if not add_comment(top_level_comment, "top_level"):
                    break

                reply_count = item.get("snippet", {}).get("totalReplyCount", 0)
                if include_replies and reply_count:
                    embedded_replies = item.get("replies", {}).get("comments", [])
                    if len(embedded_replies) == reply_count:
                        for reply in embedded_replies:
                            if not add_comment(reply, "reply"):
                                break
                    else:
                        parent_id = top_level_comment.get("id", "")
                        if parent_id and not fetch_replies(parent_id):
                            break

                if max_results is not None and len(comments) >= max_results:
                    break

            if progress_callback:
                progress_callback(len(comments))

            page_token = response.get("nextPageToken")
            if not page_token or (max_results is not None and len(comments) >= max_results):
                break
    except HttpError as error:
        LOGGER.error("YouTube API request failed: %s", error)
        raise CommentCollectionError(
            "The YouTube API request failed. Verify the API key, video visibility, "
            "comment availability, and API quota."
        ) from error
    except OSError as error:
        raise CommentCollectionError("Could not reach the YouTube API.") from error

    frame = pd.DataFrame(list(comments.values()), columns=COMMENT_COLUMNS)
    LOGGER.info("Fetched %s comments", len(frame))
    return frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect comments from a YouTube video.")
    parser.add_argument("--url", required=True, help="YouTube URL or 11-character video ID")
    parser.add_argument(
        "--api-key",
        default=os.getenv("YOUTUBE_API_KEY"),
        help="YouTube API key (defaults to YOUTUBE_API_KEY)",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=None,
        help="Maximum comments to fetch; omit to fetch all public comments.",
    )
    parser.add_argument(
        "--top-level-only",
        action="store_true",
        help="Exclude replies (by default, all public replies are included).",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/raw/youtube_comments.csv"), help="CSV output path"
    )
    return parser


def main() -> int:
    load_dotenv()
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        api_key = args.api_key or os.getenv("YOUTUBE_API_KEY", "")
        comments = get_video_comments(
            args.url,
            api_key,
            args.max_results,
            include_replies=not args.top_level_only,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        comments.to_csv(args.output, index=False)
    except (ConfigurationError, CommentCollectionError) as error:
        LOGGER.error("%s", error)
        return 1

    LOGGER.info("Saved %s comments to %s", len(comments), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
