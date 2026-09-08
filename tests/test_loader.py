"""Tests for ingest.loader."""

from pathlib import Path

import pytest

from ingest.loader import (
    Document,
    UnsupportedFormatError,
    load_document,
    load_documents,
)


def test_load_document_txt(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("hello world")
    doc = load_document(p)
    assert isinstance(doc, Document)
    assert doc.source_name == "x"
    assert doc.text == "hello world"
    assert doc.format == ".txt"
    assert doc.source_path == str(p)


def test_load_document_md(tmp_path):
    p = tmp_path / "notes.md"
    p.write_text("# Title\nbody")
    doc = load_document(p)
    assert doc.format == ".md"
    assert "Title" in doc.text


def test_load_document_pdf_creates_plain_text(tmp_path):
    # pdfplumber may not be installed in all dev envs; skip gracefully.
    pytest.importorskip("pdfplumber")
    p = tmp_path / "d.pdf"
    p.write_bytes(b"%PDF-1.4 not a real pdf")
    with pytest.raises(Exception):
        # A malformed PDF should raise (pdfplumber), not silently succeed.
        load_document(p)


def test_load_document_unsupported_extension(tmp_path):
    p = tmp_path / "doc.csv"
    p.write_text("a,b,c")
    with pytest.raises(UnsupportedFormatError):
        load_document(p)


def test_load_documents_filters_extensions(tmp_path):
    (tmp_path / "a.txt").write_text("one")
    (tmp_path / "b.md").write_text("two")
    (tmp_path / "c.pdf").write_bytes(b"%PDF")
    docs = load_documents(tmp_path, extensions={".txt"})
    assert [d.source_name for d in docs] == ["a"]


def test_load_documents_missing_dir_returns_empty(tmp_path):
    assert load_documents(tmp_path / "does-not-exist") == []


def test_load_documents_sorted(tmp_path):
    (tmp_path / "b.txt").write_text("two")
    (tmp_path / "a.txt").write_text("one")
    docs = load_documents(tmp_path)
    assert [d.source_name for d in docs] == ["a", "b"]
