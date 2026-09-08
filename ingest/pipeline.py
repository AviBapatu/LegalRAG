"""End-to-end ingestion pipeline: load -> chunk -> (SAC) -> JSONL.

Orchestrates the stages from AGENTS.md §1 and writes the result as JSONL to
`data/processed/chunks.jsonl`.
"""

from __future__ import annotations

import json
from pathlib import Path

from .chunker import Chunk, Chunker
from .loader import Document, load_documents
from .sac import apply_sac_to_document, summarize_document


def _load_config(config: dict) -> dict:
    """Extract the `chunking` and `ingest` blocks from a full config dict."""
    chunking = dict(config.get("chunking", {}))
    ingest_cfg = dict(config.get("ingest", {}))
    return {"chunking": chunking, "ingest": ingest_cfg}


def run_ingestion(
    config: dict,
    raw_dir: str | Path | None = None,
    processed_dir: str | Path | None = None,
    chunks_file: str | None = None,
    llm_func=None,
    return_chunks: bool = False,
):
    """Run the full ingestion pipeline.

    Args:
        config: Full application config dict (as loaded from config.yaml),
            containing ``chunking`` and ``ingest`` blocks.
        raw_dir: Override for the raw documents directory.
        processed_dir: Override for the output directory.
        chunks_file: Override for the output filename.
        llm_func: Optional callable used for SAC summaries when
            ``chunking.use_sac`` and ``chunking.sac_use_llm`` are both true.
        return_chunks: If True, return the list of Chunk objects in addition
            to writing the JSONL file.

    Returns:
        If ``return_chunks`` is True, a list of Chunk objects. Otherwise None.
        Always writes the JSONL output file.
    """
    cfg = _load_config(config)
    chunking_cfg = cfg["chunking"]
    ingest_cfg = cfg["ingest"]

    raw_dir = Path(raw_dir or ingest_cfg.get("raw_dir", "data/raw"))
    processed_dir = Path(processed_dir or ingest_cfg.get("processed_dir", "data/processed"))
    chunks_file = chunks_file or ingest_cfg.get("chunks_file", "chunks.jsonl")

    use_sac = bool(chunking_cfg.get("use_sac", True))
    sac_use_llm = bool(chunking_cfg.get("sac_use_llm", False))
    sac_max_chars = int(chunking_cfg.get("sac_summary_max_chars", 150))
    strategy = chunking_cfg.get("strategy", "pattern")

    chunker = Chunker(strategy, chunking_cfg.get(strategy, {}))

    documents = load_documents(raw_dir)
    all_chunks: list[Chunk] = []

    for doc in documents:
        chunks = chunker.chunk(doc.source_name, doc.text)
        if use_sac:
            summary = summarize_document(
                doc.text,
                max_chars=sac_max_chars,
                use_llm=sac_use_llm,
                llm_func=llm_func,
            ).summary
            apply_sac_to_document(chunks, summary)
        all_chunks.extend(chunks)

    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / chunks_file
    with out_path.open("w", encoding="utf-8") as fh:
        for chunk in all_chunks:
            fh.write(json.dumps(chunk.to_dict(), ensure_ascii=False))
            fh.write("\n")

    if return_chunks:
        return all_chunks
    return None
