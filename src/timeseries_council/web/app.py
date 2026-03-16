# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
FastAPI application setup for the web interface.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..logging import get_logger

logger = get_logger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="Time Series Council",
        description="AI Council for Time Series Analysis",
        version="1.0.0"
    )

    # Setup templates and static files
    web_dir = Path(__file__).parent
    templates_dir = web_dir / "templates"
    static_dir = web_dir / "static"

    # Check if we're in the new package structure or old structure
    if not templates_dir.exists():
        # Try old structure
        old_web_dir = Path(__file__).parent.parent.parent.parent / "web"
        if old_web_dir.exists():
            templates_dir = old_web_dir / "templates"
            static_dir = old_web_dir / "static"

    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
        logger.info(f"Mounted static files from {static_dir}")

    # Store templates path for routes to use
    app.state.templates_dir = templates_dir

    # Import and include routes
    from .routes import router
    app.include_router(router)

    logger.info("FastAPI application created")
    return app


# Global templates instance
_templates = None


def get_templates():
    """Get Jinja2Templates instance."""
    global _templates
    if _templates is None:
        web_dir = Path(__file__).parent
        templates_dir = web_dir / "templates"

        if not templates_dir.exists():
            old_web_dir = Path(__file__).parent.parent.parent.parent / "web"
            templates_dir = old_web_dir / "templates"

        _templates = Jinja2Templates(directory=str(templates_dir))
    return _templates


# Convenience for importing
templates = property(lambda self: get_templates())


def run_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    """Run the web server."""
    import uvicorn

    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(
        "timeseries_council.web.app:create_app",
        host=host,
        port=port,
        reload=reload,
        factory=True
    )


# Module-level app instance for direct uvicorn usage
# Usage: uvicorn timeseries_council.web.app:app
app = create_app()
