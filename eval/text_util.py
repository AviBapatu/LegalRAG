"""Shared deterministic text helpers for the Milestone 7 evaluation modules.

Tokenization, sentence splitting, token-overlap ratios, and numeric extraction
are used by ``eval/ragas_style.py`` and ``eval/claim_check.py`` so every
deterministic heuristic behaves identically and stays independently testable.

These are pure functions with no external dependencies, no network access, and
no API-key requirements. Everything here is deliberately simple and inspectable
-- they are heuristic proxies, not learned scorers.
"""

from __future__ import annotations

import re

#: Small English stopword set used to focus content tokens on meaning-bearing
#: words. Kept intentionally compact; nothing here is a model.
_STOPWORDS = frozenset(
    """
    a an the and or but if then else of for on in at by with without to from
    is are was were be been being do does did have has had will would shall
    should can could may might must not no nor this that these those it its
    their there they we you he she as so than too very under over about into
    between during per each any all both same such also more most other some
    such what which who whom whose when where why how only just
    """.split()
)

#: A single letter followed by a period: the "U.S." / "A.Key" abbreviation case.
#: A period that follows a single uppercase letter is treated as an initial
#: (part of an abbreviation), not as a sentence boundary.
_INITIAL_RE = re.compile(r"(?:^|\s)[A-Z]\.$")

#: Numeric tokens: currency/amount/percentage/ordinal-looking digit runs.
_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?%?")

#: The project's inline citation convention, ``[citation:<chunk-id>]``.
#: Citation markers are not factual claims, so they are stripped before
#: sentence splitting / claim extraction (see :func:`strip_citations`).
_CITATION_RE = re.compile(r"\[citation:[^\]]*\]", re.IGNORECASE)

#: Canonical mapping for small written-out numbers ("two" -> "2") so claims
#: that say "two years" can be checked against evidence that says "2 years".
#: The indefinite article is deliberately NOT mapped (no "a"/"an" -> 1): those
#: tokens are overwhelming the article, and mapping them produced false numeric
#: "contradictions" on ordinary sentences.
_NUMBER_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90", "hundred": "100", "thousand": "1000",
}


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric word tokens.

    ``"Confidential Information (Sec. 3.2)"`` ->
    ``["confidential", "information", "sec", "3", "2"]``.
    """
    return re.findall(r"[a-z0-9]+", str(text).lower())


def content_tokens(text: str) -> list[str]:
    """Tokens with the small stopword set removed (meaning-bearing words)."""
    return [t for t in tokenize(text) if t not in _STOPWORDS]


def content_token_set(text: str) -> set[str]:
    """Set of meaning-bearing tokens, deduplicated (order-independent)."""
    return set(content_tokens(text))


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into non-empty sentences on ``.``/``!``/``?``/``;`` and
    newlines.

    A period directly after a single uppercase letter (the ``U.S.`` /
    ``N.D.A.`` case) is NOT a boundary, so initials are kept together and legal
    abbreviations do not explode into bogus claims.
    """
    text = str(text).strip()
    if not text:
        return []

    sentences: list[str] = []
    buf = ""
    for idx, ch in enumerate(text):
        buf += ch
        prev = text[idx - 1] if idx > 0 else ""
        if ch in ".!?;" and not (ch == "." and prev.isupper()):
            piece = buf.strip()
            if piece:
                sentences.append(piece)
            buf = ""
    tail = buf.strip()
    if tail:
        sentences.append(tail)
    return sentences


def strip_citations(text: str) -> str:
    """Remove ``[citation:<id>]`` markers from text.

    Citation markers are citation plumbing, not factual content, so they are
    stripped before sentence splitting / claim extraction. A text with no
    citation markers passes through unchanged.
    """
    return _CITATION_RE.sub("", str(text))


def overlap_ratio(needle: str, haystack: str) -> float:
    """Fraction of ``needle``'s content tokens that occur in ``haystack``.

    ``0.0`` when ``needle`` has no content tokens or when no token matches;
    ``1.0`` when every meaningful ``needle`` token appears in ``haystack``.
    """
    needle_tokens = content_token_set(needle)
    if not needle_tokens:
        return 0.0
    haystack_tokens = set(tokenize(haystack))
    if not haystack_tokens:
        return 0.0
    hits = sum(1 for t in needle_tokens if t in haystack_tokens)
    return hits / len(needle_tokens)


def find_numbers(text: str) -> set[str]:
    """Canonical numeric values mentioned in ``text``.

    Returns a set of normalized number strings (commas stripped, written-out
    numbers 1..20/30..100/1000 mapped to digits). Used by the deterministic
    claim checker to detect numeric contradictions such as "two years" vs.
    "3 years".
    """
    numbers: set[str] = set()
    for raw in _NUMBER_RE.findall(str(text)):
        number = raw.replace(",", "").replace("%", "")
        if number:
            numbers.add(number)
    for word in str(text).lower().split():
        mapped = _NUMBER_WORDS.get(word.strip(".,;!?"))
        if mapped:
            numbers.add(mapped)
    return numbers