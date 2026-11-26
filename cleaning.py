"""Conservative task text normalization for customer and social text."""

from __future__ import annotations

import html
import re
import unicodedata

import pandas as pd

_HTML_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")


def clean_text(value: object) -> str:
    """Normalize markup and spacing while retaining negation, emoji and punctuation.

    URLs and HTML tags are removed, but mentions, hashtags, contractions, casing,
    punctuation and emoji remain available as sentiment-bearing text.
    """
    if value is None:
        return ""
    missing = pd.isna(value)
    if isinstance(missing, bool) and missing:
        return ""
    text = html.unescape(str(value))
    text = unicodedata.normalize("NFKC", text)
    text = _HTML_TAG.sub(" ", text)
    text = _URL.sub(" <URL> ", text)
    return _WHITESPACE.sub(" ", text).strip()
