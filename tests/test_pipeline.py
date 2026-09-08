"""Tests for ingest.pipeline: end-to-end ingestion -> JSONL."""

import json
from pathlib import Path

import pytest

from ingest.pipeline import run_ingestion


def _config(**overrides):
    cfg = {
        "ingest": {
            "raw_dir": "data/raw",
            "processed_dir": "data/processed",
            "chunks_file": "chunks.jsonl",
        },
        "chunking": {
            "strategy": "pattern",
            "use_sac": True,
            "sac_use_llm": False,
            "sac_summary_max_chars": 150,
            "pattern": {},
            "sentence": {},
        },
    }
    cfg.update(overrides)
    return cfg


def _write_doc(directory: Path, name: str, text: str):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(text)


SAMPLE_DOC = """Section 1. First Part
Some body text here.

Section 2. Second Part
2.1 A subsection body.
"""


def test_run_ingestion_writes_jsonl(tmp_path):
    raw = tmp_path / "raw"
    proc = tmp_path / "processed"
    _write_doc(raw, "a.txt", SAMPLE_DOC)
    cfg = _config(
        ingest={
            "raw_dir": str(raw),
            "processed_dir": str(proc),
            "chunks_file": "chunks.jsonl",
        }
    )
    run_ingestion(cfg)
    out = proc / "chunks.jsonl"
    assert out.exists()
    lines = out.read_text().strip().splitlines()
    assert len(lines) >= 2
    first = json.loads(lines[0])
    assert "chunk_id" in first
    assert "source_name" in first
    assert "text" in first
    assert "original_text" in first
    assert "section" in first
    assert "parent" in first
    assert "heading" in first
    assert "hierarchy_path" in first
    assert "start_offset" in first
    assert "end_offset" in first
    assert "sac_applied" in first


def test_run_ingestion_sac_on_by_default(tmp_path):
    raw = tmp_path / "raw"
    proc = tmp_path / "processed"
    _write_doc(raw, "a.txt", SAMPLE_DOC)
    cfg = _config(ingest={"raw_dir": str(raw), "processed_dir": str(proc)})
    run_ingestion(cfg)
    lines = (proc / "chunks.jsonl").read_text().splitlines()
    chunked = [json.loads(l) for l in lines if l.strip()]
    assert all(c["sac_applied"] is True for c in chunked)
    assert all("[Document summary]" in c["text"] for c in chunked)


def test_run_ingestion_sac_off(tmp_path):
    raw = tmp_path / "raw"
    proc = tmp_path / "processed"
    _write_doc(raw, "a.txt", SAMPLE_DOC)
    cfg = _config(
        ingest={"raw_dir": str(raw), "processed_dir": str(proc)},
        chunking={"strategy": "pattern", "use_sac": False},
    )
    run_ingestion(cfg)
    chunked = [json.loads(l) for l in (proc / "chunks.jsonl").read_text().splitlines() if l.strip()]
    assert all(c["sac_applied"] is False for c in chunked)
    assert all("[Document summary]" not in c["text"] for c in chunked)


def test_run_ingestion_empty_raw_dir(tmp_path):
    raw = tmp_path / "empty"
    proc = tmp_path / "processed"
    raw.mkdir()
    cfg = _config(ingest={"raw_dir": str(raw), "processed_dir": str(proc)})
    run_ingestion(cfg)
    assert (proc / "chunks.jsonl").exists()
    assert (proc / "chunks.jsonl").read_text().strip() == ""


def test_run_ingestion_return_chunks(tmp_path):
    raw = tmp_path / "raw"
    proc = tmp_path / "processed"
    _write_doc(raw, "a.txt", SAMPLE_DOC)
    cfg = _config(
        ingest={"raw_dir": str(raw), "processed_dir": str(proc)},
        chunking={"strategy": "pattern", "use_sac": False},
    )
    chunks = run_ingestion(cfg, return_chunks=True)
    assert chunks
    assert all(c.source_name == "a" for c in chunks)


def test_run_ingestion_llm_summary(tmp_path):
    raw = tmp_path / "raw"
    proc = tmp_path / "processed"
    _write_doc(raw, "a.txt", SAMPLE_DOC)

    def fake_llm(text):
        return "FINGERPRINT"

    cfg = _config(
        ingest={"raw_dir": str(raw), "processed_dir": str(proc)},
        chunking={
            "strategy": "pattern",
            "use_sac": True,
            "sac_use_llm": True,
        },
    )
    run_ingestion(cfg, llm_func=fake_llm)
    chunked = [json.loads(l) for l in (proc / "chunks.jsonl").read_text().splitlines() if l.strip()]
    assert all("FINGERPRINT" in c["text"] for c in chunked)
