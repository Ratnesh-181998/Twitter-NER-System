"""Normalize saved Xquik tweet exports for the Streamlit NER app."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

TEXT_FIELDS = ("tweet_text", "full_text", "text", "content", "body")
ID_FIELDS = ("tweet_id", "id", "post_id")
AUTHOR_FIELDS = ("username", "author_username", "screen_name", "user")
CREATED_FIELDS = ("created_at", "timestamp", "date")


def _first_value(row: Mapping[str, Any], fields: Iterable[str]) -> str:
    for field in fields:
        value = row.get(field)
        if value is None:
            continue

        text = str(value).strip()
        if text:
            return text

    return ""


def normalize_xquik_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Convert Xquik export rows into text records ready for NER analysis."""
    normalized: list[dict[str, str]] = []

    for row in rows:
        text = _first_value(row, TEXT_FIELDS)
        if not text:
            continue

        normalized.append(
            {
                "text": text,
                "tweet_id": _first_value(row, ID_FIELDS),
                "author": _first_value(row, AUTHOR_FIELDS),
                "created_at": _first_value(row, CREATED_FIELDS),
            }
        )

    return normalized


def summarize_xquik_rows(rows: Iterable[Mapping[str, str]]) -> dict[str, int]:
    records = list(rows)
    return {
        "rows": len(records),
        "with_author": sum(1 for row in records if row.get("author")),
        "with_created_at": sum(1 for row in records if row.get("created_at")),
    }
