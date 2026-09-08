"""Tests for config loading and validation (config.yaml)."""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config.yaml"


def test_config_exists():
    assert CONFIG_PATH.exists()


def test_config_is_valid_yaml_with_expected_structure():
    with CONFIG_PATH.open() as fh:
        cfg = yaml.safe_load(fh)
    assert "ingest" in cfg
    assert "chunking" in cfg
    assert "output" in cfg
    # Required keys for Milestone 1
    assert "use_sac" in cfg["chunking"]
    assert cfg["chunking"]["use_sac"] in (True, False)
    assert cfg["chunking"]["strategy"] in ("sentence", "pattern")
    assert "chunks_file" in cfg["ingest"]
    assert cfg["ingest"]["chunks_file"].endswith(".jsonl")
    assert "raw_dir" in cfg["ingest"]
    assert "processed_dir" in cfg["ingest"]
