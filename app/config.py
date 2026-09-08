"""Configuration for the Milestone 8 API layer.

Only settings genuinely required by the API are added here (AGENTS.md §8):
the bind host/port used by the dev uvicorn launcher and the path to the
evaluation result file. All Milestone 1-7 settings live in the shared
``config.yaml`` and are loaded by the app at construction time; nothing in this
module replaces or duplicates them. No secrets are stored here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class AppSettings:
    """Runtime settings for the FastAPI layer itself.

    Note: these are independent of the project's ``config.yaml`` (which we
    never rewrite). They add only the small amount of API-specific wiring the
    spec calls for.
    """

    host: str = "127.0.0.1"
    port: int = 8000
    #: Path to the Milestone 7 full-evaluation results JSON. Once a run is
    #: committed or cached, ``/eval`` reads this file rather than re-running.
    eval_results_path: str = "eval/results/full_eval/results.json"
    #: Path to the project config.yaml (Milestone 1-7).
    config_path: str = "config.yaml"

    @classmethod
    def from_mapping(cls, mapping: Mapping | None) -> "AppSettings":
        mapping = mapping or {}
        app_block = mapping.get("app") or {}
        return cls(
            host=str(app_block.get("host", cls.host)),
            port=int(app_block.get("port", cls.port)),
            eval_results_path=str(
                mapping.get("eval_results_path", cls.eval_results_path)
            ),
            config_path=str(mapping.get("config_path", cls.config_path)),
        )

    @classmethod
    def with_override(
        cls, base: "AppSettings", **overrides: object
    ) -> "AppSettings":
        data = {
            "host": base.host,
            "port": base.port,
            "eval_results_path": base.eval_results_path,
            "config_path": base.config_path,
        }
        data.update(overrides)
        return cls(**data)
