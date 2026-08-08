import pytest

from data_collector import ConfigurationError, extract_video_id, get_video_comments


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("kqtD5dpn9C8", "kqtD5dpn9C8"),
        ("https://www.youtube.com/watch?v=kqtD5dpn9C8", "kqtD5dpn9C8"),
        ("https://youtu.be/kqtD5dpn9C8?t=10", "kqtD5dpn9C8"),
        ("https://www.youtube.com/shorts/kqtD5dpn9C8", "kqtD5dpn9C8"),
    ],
)
def test_extract_video_id_accepts_supported_forms(value, expected):
    assert extract_video_id(value) == expected


def test_extract_video_id_rejects_invalid_values():
    with pytest.raises(ConfigurationError):
        extract_video_id("https://example.com/not-a-video")


class _Request:
    def execute(self):
        return {
            "items": [
                {
                    "snippet": {
                        "topLevelComment": {
                            "id": "comment-1",
                            "snippet": {
                                "authorDisplayName": "A viewer",
                                "textDisplay": "Useful tutorial",
                                "likeCount": 3,
                                "publishedAt": "2025-01-01T00:00:00Z",
                                "updatedAt": "2025-01-01T00:00:00Z",
                            },
                        }
                    }
                },
                {
                    "snippet": {
                        "topLevelComment": {
                            "id": "comment-2",
                            "snippet": {"textDisplay": "A second comment"},
                        }
                    }
                },
            ]
        }


class _CommentThreads:
    def __init__(self):
        self.request_arguments = None

    def list(self, **kwargs):
        self.request_arguments = kwargs
        return _Request()


class _YoutubeClient:
    def __init__(self):
        self.threads = _CommentThreads()

    def commentThreads(self):
        return self.threads


def test_collection_honours_requested_maximum_and_schema():
    client = _YoutubeClient()
    comments = get_video_comments("kqtD5dpn9C8", "unused", max_results=1, youtube_client=client)

    assert len(comments) == 1
    assert comments.loc[0, "comment_id"] == "comment-1"
    assert client.threads.request_arguments["maxResults"] == 1


def _comment(comment_id: str, text: str, parent_id: str = "") -> dict:
    snippet = {
        "authorDisplayName": "A viewer",
        "textDisplay": text,
        "likeCount": 0,
        "publishedAt": "2025-01-01T00:00:00Z",
        "updatedAt": "2025-01-01T00:00:00Z",
    }
    if parent_id:
        snippet["parentId"] = parent_id
    return {"id": comment_id, "snippet": snippet}


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def execute(self):
        return self.payload


class _PagedThreads:
    def __init__(self, pages):
        self.pages = pages
        self.calls = []

    def list(self, **kwargs):
        self.calls.append(kwargs)
        return _Response(self.pages[kwargs.get("pageToken")])


class _Replies:
    def __init__(self, pages):
        self.pages = pages
        self.calls = []

    def list(self, **kwargs):
        self.calls.append(kwargs)
        return _Response(self.pages[(kwargs["parentId"], kwargs.get("pageToken"))])


class _PagedYoutubeClient:
    def __init__(self, thread_pages, reply_pages=None):
        self.threads = _PagedThreads(thread_pages)
        self.replies = _Replies(reply_pages or {})

    def commentThreads(self):
        return self.threads

    def comments(self):
        return self.replies


def test_collection_without_a_limit_follows_every_comment_thread_page():
    client = _PagedYoutubeClient(
        {
            None: {
                "items": [{"snippet": {"topLevelComment": _comment("first", "First")}}],
                "nextPageToken": "page-2",
            },
            "page-2": {"items": [{"snippet": {"topLevelComment": _comment("second", "Second")}}]},
        }
    )

    comments = get_video_comments(
        "kqtD5dpn9C8", "unused", max_results=None, include_replies=False, youtube_client=client
    )

    assert comments["comment_id"].tolist() == ["first", "second"]
    assert [call["maxResults"] for call in client.threads.calls] == [100, 100]


def test_collection_fetches_complete_reply_pages_when_thread_replies_are_partial():
    parent = _comment("parent", "Top-level")
    client = _PagedYoutubeClient(
        {None: {"items": [{"snippet": {"topLevelComment": parent, "totalReplyCount": 2}}]}},
        {
            ("parent", None): {
                "items": [_comment("reply-1", "One", "parent")],
                "nextPageToken": "more-replies",
            },
            ("parent", "more-replies"): {"items": [_comment("reply-2", "Two", "parent")]},
        },
    )

    comments = get_video_comments("kqtD5dpn9C8", "unused", youtube_client=client)

    assert comments["comment_id"].tolist() == ["parent", "reply-1", "reply-2"]
    assert comments["comment_type"].tolist() == ["top_level", "reply", "reply"]
    assert len(client.replies.calls) == 2
