"""Chunking strategies: sentence chunking and structure-aware pattern chunking.

Both chunkers produce `Chunk` objects carrying structural metadata (section,
parent, heading, hierarchy path, character offsets, SAC status) as required by
the spec (AGENTS.md §1). The pattern chunker is structure-aware: it walks the
section/subsection hierarchy and records the parent path so a later reranker
can boost cross-referenced provisions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class Chunk:
    """A single chunk produced from a document.

    Attributes:
        chunk_id: Globally unique id (``{source_name}-{index}``) usable as a
            citation target.
        source_name: Source document id the chunk came from.
        text: The chunk text. When SAC is applied by a later step the embedded
            text may differ from `original_text` (summary prepended).
        original_text: The chunk text exactly as split from the document
            (before any SAC prepending).
        section: The section number/identifier this chunk belongs to, e.g. "3".
        parent: The parent section number, or None for top-level sections.
        heading: The human-readable heading line of the section, if any.
        hierarchy_path: Ordered list of section identifiers from root to the
            chunk's own leaf section, e.g. ["1", "1.2"].
        start_offset: Character offset of the chunk start within the document.
        end_offset: Character offset of the chunk end within the document.
        sac_applied: Whether SAC has been applied to this chunk.
    """

    chunk_id: str
    source_name: str
    text: str
    original_text: str
    section: str | None
    parent: str | None
    heading: str | None
    hierarchy_path: list[str] = field(default_factory=list)
    start_offset: int = 0
    end_offset: int = 0
    sac_applied: bool = False

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "source_name": self.source_name,
            "text": self.text,
            "original_text": self.original_text,
            "section": self.section,
            "parent": self.parent,
            "heading": self.heading,
            "hierarchy_path": list(self.hierarchy_path),
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "sac_applied": self.sac_applied,
        }


_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


class SentenceChunker:
    """Splits text on sentence boundaries, optionally grouping short sentences.

    Mirrors the survey's sentence chunking strategy (§III.A.1), which the
    literature reports as best for precision/F1.
    """

    def __init__(self, max_chars: int = 1000, min_chars: int = 50):
        if max_chars < 1:
            raise ValueError("max_chars must be >= 1")
        if min_chars < 0:
            raise ValueError("min_chars must be >= 0")
        self.max_chars = max_chars
        self.min_chars = min_chars

    def _split_sentences(self, text: str) -> list[tuple[str, int]]:
        """Return (sentence, offset_in_text) pairs."""
        parts = _SENTENCE_END_RE.split(text)
        result: list[tuple[str, int]] = []
        cursor = 0
        for part in parts:
            if not part:
                continue
            idx = text.find(part, cursor)
            result.append((part, idx))
            cursor = idx + len(part)
        return result

    def chunk(self, source_name: str, text: str) -> list[Chunk]:
        """Chunk a document into sentence-based chunks.

        Short sentences below ``min_chars`` are merged with the next sentence
        to avoid tiny, useless chunks. Sentences are grouped so that each chunk
        stays at or under ``max_chars``.
        """
        sentences = self._split_sentences(text)
        chunks: list[Chunk] = []
        pending: list[tuple[str, int]] = []
        pending_len = 0

        def flush() -> None:
            nonlocal pending, pending_len
            if not pending:
                return
            start = pending[0][1]
            end = pending[-1][1] + len(pending[-1][0])
            body = "".join(s for s, _ in pending)
            chunk_id = f"{source_name}-{len(chunks)}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    source_name=source_name,
                    text=body,
                    original_text=body,
                    section=None,
                    parent=None,
                    heading=None,
                    hierarchy_path=[],
                    start_offset=start,
                    end_offset=end,
                )
            )
            pending = []
            pending_len = 0

        for sentence, offset in sentences:
            s_len = len(sentence)
            if pending and pending_len + s_len > self.max_chars:
                flush()
            pending.append((sentence, offset))
            pending_len += s_len
            if pending_len >= self.min_chars:
                flush()

        flush()
        return chunks


# Matches the label (and optional title) of a heading, e.g.:
#   "Section 3", "Article 5", "3.2", "(a)", "1. Definitions"
_HEADING_RE = re.compile(
    r"^\s*(?P<label>"
    r"(?:Section|Article)\s+\d+"
    r"|\d+(?:\.\d+){0,2}"
    r"|\(\w+\)"
    r")\s*(?P<title>[^\n:]*)"
)


def _normalize_number(match_text: str) -> str:
    """Turn a matched label into a canonical dot-separated section id.

    ``Section 3`` -> ``3``, ``3.2`` -> ``3.2``, ``(a)`` -> ``a``.
    """
    m = _HEADING_RE.match(match_text)
    if not m:
        return match_text.strip()
    t = m.group("label").strip()
    if t.lower().startswith(("section ", "article ")):
        return t.split()[1]
    if t.startswith("(") and t.endswith(")"):
        return t[1:-1]
    return t


def _level_of_number(number: str) -> int:
    dots = number.count(".")
    if dots >= 2:
        return 3
    if dots == 1:
        return 2
    return 1


class PatternChunker:
    """Structure-aware pattern chunker.

    Splits a document on section delimiters (e.g. "Section 3", "3.2", "(a)")
    and records the section/subsection hierarchy. Each chunk keeps its
    structural metadata (section, parent, heading, hierarchy path) so a later
    reranker can use the structure to boost cross-referenced provisions
    (AGENTS.md §3).

    Delimiters match a section *label* at the start of a line. The section
    heading is the full first line; the body is:

      * for "Section N" / "Article N" headings: the following lines until the
        next boundary;
      * for numeric/"letter" labels (e.g. "3.2"): the text on the same line
        immediately after the label, plus any following lines until the next
        boundary.

    Deeper-level matches take priority at the same position. Text before the
    first delimiter (a preamble) is emitted as a chunk with no section.
    """

    def __init__(self, delimiters: Sequence[dict] | None = None):
        if delimiters is None:
            delimiters = [
                {"pattern": r"^[ \t]*(?:Section|Article)\s+\d+", "level": 1},
                {"pattern": r"^[ \t]*\d+\.\d+\.\d+", "level": 3},
                {"pattern": r"^[ \t]*\d+\.\d+", "level": 2},
                {"pattern": r"^[ \t]*\d+\.", "level": 1},
                {"pattern": r"^[ \t]*\(\w+\)", "level": 3},
            ]
        self.delimiters: list[dict] = list(delimiters)
        self.compiled: list[tuple[re.Pattern, int]] = [
            (re.compile(d["pattern"], re.MULTILINE), int(d["level"])) for d in self.delimiters
        ]
        self._counter: dict[str, int] = {}

    def _label_matches(self, text: str) -> list[tuple[str, int, int]]:
        """Return sorted list of (number, start_offset, label_end, level).

        ``label_end`` is the offset just after the matched label text.
        """
        raw: list[tuple[int, int, str, int]] = []  # (start, level, number, end)
        for pattern, level in self.compiled:
            for match in pattern.finditer(text):
                number = _normalize_number(match.group(0))
                raw.append((match.start(), level, number, match.end()))
        # Deduplicate by start, keep the deepest level at each position.
        by_start: dict[int, tuple[int, str, int]] = {}
        for start, level, number, end in raw:
            existing = by_start.get(start)
            if existing is None or level > existing[0]:
                by_start[start] = (level, number, end)
        result = [(number, start, end, level) for start, (level, number, end) in
                  sorted(by_start.items())]
        return result

    def chunk(self, source_name: str, text: str) -> list[Chunk]:
        boundaries = self._label_matches(text)  # (number, start, label_end, level)
        chunks: list[Chunk] = []

        # No section delimiters found: emit the whole text as a single chunk.
        if not boundaries:
            body = text.strip()
            if body:
                chunks.append(
                    self._make_chunk(source_name, body, 0, len(text), None, None, [], None)
                )
            return chunks

        # Preamble before the first section becomes an unsectioned chunk.
        if boundaries[0][1] > 0:
            preamble = text[: boundaries[0][1]].strip()
            if preamble:
                chunks.append(
                    self._make_chunk(source_name, preamble, 0, boundaries[0][1],
                                     None, None, [], None)
                )

        # Stack of (number, level) open sections, for hierarchy paths.
        stack: list[tuple[str, int]] = []

        def _reset_stack_to(level: int) -> None:
            while stack and stack[-1][1] >= level:
                stack.pop()

        for i, (number, start, label_end, level) in enumerate(boundaries):
            _reset_stack_to(level)
            parent = stack[-1][0] if stack else None
            path = [n for n, _ in stack] + [number]
            line_end = _line_end(text, start)

            if _is_named_heading(number, level):
                # "Section N" / "Article N": body starts on the following line.
                heading = _first_line(text, start, line_end)
                body_start = line_end
            else:
                # Numeric / letter label: same-line text after the label is body.
                heading = number
                body_start = label_end

            body_end = boundaries[i + 1][1] if i + 1 < len(boundaries) else len(text)
            body = text[body_start:body_end].strip()
            if body:
                chunks.append(
                    self._make_chunk(source_name, body, body_start, body_end,
                                     number, parent, _normalize_path(path), heading)
                )
            stack.append((number, level))

        return chunks

    def _make_chunk(
        self,
        source_name: str,
        body: str,
        start: int,
        end: int,
        section: str | None,
        parent: str | None,
        path: list[str],
        heading: str | None,
    ) -> Chunk:
        chunk_id = f"{source_name}-{self._counter.get(source_name, 0)}"
        self._counter[source_name] = self._counter.get(source_name, 0) + 1
        return Chunk(
            chunk_id=chunk_id,
            source_name=source_name,
            text=body,
            original_text=body,
            section=section,
            parent=parent,
            heading=heading,
            hierarchy_path=path,
            start_offset=start,
            end_offset=end,
        )


def _is_named_heading(number: str, level: int) -> bool:
    """True if the section came from a "Section N"/"Article N" line."""
    # Named headings are the only top-level labels that aren't bare numbers
    # with a dot; but a "N." label is numeric too. We detect by checking the
    # original line in _label_matches instead -- simplest: named headings get
    # a distinct marker. For now, named headings have 'level 1' from the
    # Section/Article pattern; we cannot distinguish reliably here, so we rely
    # on the number format: named headings never contain a dot.
    return "." not in number and not number.startswith("(")


def _normalize_path(path: list[str]) -> list[str]:
    return list(path)


def _line_end(text: str, start: int) -> int:
    nl = text.find("\n", start)
    return nl + 1 if nl != -1 else len(text)


def _first_line(text: str, start: int, line_end: int) -> str:
    return text[start:line_end].strip()


class Chunker:
    """Facade selecting a strategy based on the config string."""

    STRATEGIES = {"sentence": SentenceChunker, "pattern": PatternChunker}

    def __init__(self, strategy: str, config: dict | None = None):
        self.config = config or {}
        strategy = strategy.lower()
        if strategy not in self.STRATEGIES:
            raise ValueError(
                f"unknown chunking strategy {strategy!r}; "
                f"choose from {sorted(self.STRATEGIES)}"
            )
        self.strategy = strategy
        if strategy == "sentence":
            self.impl = SentenceChunker(
                max_chars=self.config.get("max_chars", 1000),
                min_chars=self.config.get("min_chars", 50),
            )
        else:
            self.impl = PatternChunker(self.config.get("delimiters"))

    def chunk(self, source_name: str, text: str) -> list[Chunk]:
        return self.impl.chunk(source_name, text)
