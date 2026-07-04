import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "streamlit_deploy"))

from xquik_export import normalize_xquik_rows, summarize_xquik_rows


class XquikExportTest(unittest.TestCase):
    def test_normalizes_common_xquik_columns(self):
        rows = [
            {
                "tweet_id": "123",
                "tweet_text": "Apple opened a new store in London.",
                "username": "brandwatch",
                "created_at": "2026-07-04T10:00:00Z",
            },
            {"full_text": "   "},
            {"content": "Tesla ships Model Y in Berlin.", "screen_name": "news"},
        ]

        normalized = normalize_xquik_rows(rows)

        self.assertEqual(
            normalized,
            [
                {
                    "text": "Apple opened a new store in London.",
                    "tweet_id": "123",
                    "author": "brandwatch",
                    "created_at": "2026-07-04T10:00:00Z",
                },
                {
                    "text": "Tesla ships Model Y in Berlin.",
                    "tweet_id": "",
                    "author": "news",
                    "created_at": "",
                },
            ],
        )

    def test_summarizes_available_metadata(self):
        summary = summarize_xquik_rows(
            [
                {"text": "one", "author": "alice", "created_at": "2026-07-04"},
                {"text": "two", "author": "", "created_at": ""},
            ]
        )

        self.assertEqual(summary, {"rows": 2, "with_author": 1, "with_created_at": 1})


if __name__ == "__main__":
    unittest.main()
