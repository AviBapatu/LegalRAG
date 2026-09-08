"""FastAPI route handlers (Milestone 8).

These handlers are deliberately thin: they validate the HTTP request, delegate
to the service-layer ``Application``, and translate service exceptions into
HTTP errors. All business logic lives in ``app.service``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse

from .config import AppSettings
from .schemas import QueryRequest
from .service import (
    AppError,
    Application,
    ConfigNotFoundError,
    EvalResultsNotFoundError,
    GenerationError,
    InvalidQueryError,
    RetrievalError,
)

router = APIRouter()

_STATIC_DIR = Path(__file__).resolve().parent / "static"


def _error_to_http(exc: AppError, status: int) -> HTTPException:
    return HTTPException(status_code=status, detail=str(exc))


def get_application(request: Request) -> Application:
    """FastAPI dependency: retrieve the shared Application from app.state.

    Tests and callers inject an ``Application`` into ``app.state.app``.
    """
    return request.app.state.app


@router.get("/")
def serve_frontend(request: Request):
    """Serve the single-page frontend."""
    index_path = _STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="frontend not found")
    return FileResponse(str(index_path), media_type="text/html")


@router.post("/query")
def query(
    body: QueryRequest,
    app: Application = Depends(get_application),
) -> dict[str, Any]:
    """Run a query through the adaptive RAG pipeline and return structured info.

    Response is a flat dict (the QueryResponse model keeps the schema
    documented but serialization is intentionally dict-based for flexibility).
    """
    query_text = (body.query or "").strip()
    if not query_text:
        raise HTTPException(status_code=422, detail="query must be a non-empty string")

    try:
        result = app.answer_query(query_text)
    except InvalidQueryError as exc:
        raise _error_to_http(exc, 422)
    except RetrievalError as exc:
        raise _error_to_http(exc, 503)
    except GenerationError as exc:
        raise _error_to_http(exc, 503)
    except AppError as exc:
        raise _error_to_http(exc, 500)
    return result


@router.get("/eval")
def eval_summary(
    app: Application = Depends(get_application),
) -> dict[str, Any]:
    """Return the latest available Milestone 7 evaluation summary."""
    try:
        return app.eval_summary()
    except EvalResultsNotFoundError as exc:
        raise _error_to_http(exc, 404)
    except ConfigNotFoundError as exc:
        raise _error_to_http(exc, 500)
    except AppError as exc:
        raise _error_to_http(exc, 500)
