"""Load raw legal documents (PDF/TXT/MD) from a directory into Document objects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


class UnsupportedFormatError(ValueError):
    """Raised when a file extension is not a supported document format."""


@dataclass(frozen=True)
class Document:
    """A single loaded source document.

    Attributes:
        source_path: Absolute path to the original file.
        source_name: Stable identifier, e.g. the file stem. Used as the document
            id throughout the pipeline.
        text: The full plain-text content of the document.
        format: The file extension (".pdf", ".txt", or ".md").
    """

    source_path: str
    source_name: str
    text: str
    format: str


def _read_text(path: Path) -> str:
    """Read a plain-text (.txt/.md) file, guessing an encoding."""
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"could not decode {path.name} with any supported encoding")


def _read_pdf(path: Path) -> str:
    """Extract text from a PDF using pdfplumber."""
    import pdfplumber

    pages: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages.append(text)
    return "\n\n".join(pages)


def load_document(path: str | Path) -> Document:
    """Load a single file into a Document, dispatching on extension."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFormatError(
            f"unsupported format {ext!r} for {path.name}; "
            f"supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    if ext == ".pdf":
        text = _read_pdf(path)
    else:
        text = _read_text(path)
    return Document(
        source_path=str(path),
        source_name=path.stem,
        text=text,
        format=ext,
    )


def load_documents(
    raw_dir: str | Path,
    extensions: set[str] | None = None,
) -> list[Document]:
    """Load all supported documents from ``raw_dir`` (non-recursive).

    Args:
        raw_dir: Directory containing raw documents.
        extensions: Optional filter of extensions to load. Defaults to all
            supported extensions.

    Returns:
        A list of Document objects, one per readable file. Files that fail to
        load are skipped (the caller can detect this by comparing lengths).
        If the raw directory does not exist or is empty, an empty list is
        returned.
    """
    raw_dir = Path(raw_dir)
    extensions = extensions or SUPPORTED_EXTENSIONS
    extensions = {ext.lower() for ext in extensions}
    if not raw_dir.is_dir():
        return []

    documents: list[Document] = []
    for path in sorted(raw_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in extensions:
            continue
        documents.append(load_document(path))
    return documents
