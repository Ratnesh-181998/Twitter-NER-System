import csv
import json
from io import StringIO
from typing import Any, List, Optional

TEXT_FIELDS = (
    "text",
    "tweet_text",
    "tweetText",
    "full_text",
    "fullText",
    "content",
    "body",
    "caption",
)
NESTED_RECORD_FIELDS = ("tweet", "post", "status")
NESTED_LIST_FIELDS = ("tweets", "data", "items", "results")


def parse_tweetclaw_upload(uploaded_file) -> List[str]:
    """Extract tweet text rows from a TweetClaw JSON, JSONL, or CSV export."""
    filename = getattr(uploaded_file, "name", "").lower()
    raw_content = uploaded_file.read()

    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)

    content = raw_content.decode("utf-8-sig")

    if filename.endswith(".csv"):
        records = list(csv.DictReader(StringIO(content)))
        return _extract_texts(records)

    if filename.endswith(".jsonl"):
        records = [json.loads(line) for line in content.splitlines() if line.strip()]
        return _extract_texts(records)

    payload = json.loads(content)
    return _extract_texts(_flatten_records(payload))


def _flatten_records(payload: Any) -> List[Any]:
    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        for field in NESTED_LIST_FIELDS:
            records = payload.get(field)
            if isinstance(records, list):
                return records

        return [payload]

    return []


def _extract_texts(records: List[Any]) -> List[str]:
    texts = []

    for record in records:
        text = _extract_text(record)
        if text:
            texts.append(text)

    return texts


def _extract_text(record: Any) -> Optional[str]:
    if not isinstance(record, dict):
        return None

    for field in TEXT_FIELDS:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()

    for field in NESTED_RECORD_FIELDS:
        nested = record.get(field)
        if isinstance(nested, dict):
            text = _extract_text(nested)
            if text:
                return text

    return None
