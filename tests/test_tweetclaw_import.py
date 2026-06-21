import json
import unittest
from io import BytesIO

from streamlit_deploy.tweetclaw_import import parse_tweetclaw_upload


class NamedBytesIO(BytesIO):
    def __init__(self, name, content):
        super().__init__(content)
        self.name = name


class TweetClawImportTest(unittest.TestCase):
    def test_extracts_text_from_json_export(self):
        payload = {
            "tweets": [
                {"tweet": {"text": "Apple launched a new MacBook in London."}},
                {"tweet": {"full_text": "Tesla opened a factory in Berlin."}},
            ]
        }
        upload = NamedBytesIO("tweetclaw.json", json.dumps(payload).encode("utf-8"))

        texts = parse_tweetclaw_upload(upload)

        self.assertEqual(
            texts,
            [
                "Apple launched a new MacBook in London.",
                "Tesla opened a factory in Berlin.",
            ],
        )

    def test_extracts_text_from_jsonl_export(self):
        upload = NamedBytesIO(
            "tweetclaw.jsonl",
            b'{"text":"Nvidia announced Blackwell."}\n{"content":"Google opened an office."}\n',
        )

        texts = parse_tweetclaw_upload(upload)

        self.assertEqual(
            texts,
            ["Nvidia announced Blackwell.", "Google opened an office."],
        )

    def test_extracts_text_from_csv_export(self):
        upload = NamedBytesIO(
            "tweetclaw.csv",
            b"id,tweet_text\n1,Microsoft announced Copilot in Seattle.\n",
        )

        texts = parse_tweetclaw_upload(upload)

        self.assertEqual(texts, ["Microsoft announced Copilot in Seattle."])


if __name__ == "__main__":
    unittest.main()
