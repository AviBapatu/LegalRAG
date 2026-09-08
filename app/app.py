"""FastAPI application factory (Milestone 8, AGENTS.md §7/§8).

``create_app`` builds a FastAPI app whose ``state.app`` holds the service-layer
``Application``. The factory is cheap: constructing the app does NOT build a
real retriever, generator, adaptive controller, or Groq client — those are
only constructed lazily when ``/query`` is actually served with no injected
components present. Tests and local smoke runs inject fake components, so no
API key, network access, or embedding-model download is ever required to import
or to test this package.
"""

from __future__ import annotations

from typing import Mapping

from fastapi import FastAPI

from .config import AppSettings
from .routes import router
from .service import Application


def create_app(
    settings: AppSettings | None = None,
    *,
    app_: Application | None = None,
    config: Mapping | None = None,
) -> FastAPI:
    """Build the FastAPI application.

    Args:
        settings: Optional :class:`AppSettings`; defaults to AppSettings().
        app_: An already-constructed service :class:`Application`. Use this to
            attach a custom/dependency-injected app for tests.
        config: Optional in-memory project config mapping. Passed through to
            the Application when ``app_`` is not supplied.

    Returns:
        A FastAPI ``FastAPI`` instance wired with the Milestone 8 routes.
    """
    settings = settings or AppSettings()
    if app_ is None:
        app_ = Application(settings=settings, config=config)

    fastapi_app = FastAPI(
        title="LegalRAG",
        description=(
            "Retrieval-augmented generation for legal documents "
            "(Hindi et al. reference implementation)."
        ),
        version="0.1.0",
    )
    fastapi_app.state.app = app_
    fastapi_app.state.settings = settings
    fastapi_app.include_router(router)
    return fastapi_app
