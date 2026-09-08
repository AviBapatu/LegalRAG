"""Run the LegalRAG Milestone 8 API locally (dev launcher).

Usage (from the repo root):

    python -m app            # default: 127.0.0.1:8000
    python -m app --host 0.0.0.0 --port 8080

No API key is needed to *start* the server. A real Groq call (and therefore
a key) is only required when a query actually needs the real generator.
"""

from __future__ import annotations

import argparse
import sys

from .app import create_app
from .config import AppSettings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LegalRAG Milestone 8 API")
    parser.add_argument("--host", default=None, help="bind host (default from config)")
    parser.add_argument("--port", default=None, type=int, help="bind port (default from config)")
    args = parser.parse_args(argv)

    settings = AppSettings()
    if args.host:
        settings = AppSettings.with_override(settings, host=args.host)
    if args.port:
        settings = AppSettings.with_override(settings, port=args.port)

    app = create_app(settings=settings)

    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
