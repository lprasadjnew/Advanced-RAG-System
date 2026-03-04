"""
routes_pages.py — Landing page, Terms of Use, and Privacy Policy endpoints.

Add to your existing FastAPI app:

    from routes_pages import register_page_routes
    register_page_routes(app)

Place the static/ folder (with index.html, termsofuse.html, privacy.html)
in the same directory as this file.
"""

import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


def _read_html(filename: str) -> str:
    """Read an HTML file from the static directory."""
    filepath = os.path.join(_STATIC_DIR, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def register_page_routes(app: FastAPI):
    """Register the landing page, terms, and privacy routes."""

    @app.get("/", response_class=HTMLResponse, tags=["Pages"])
    async def landing_page():
        return _read_html("index.html")

    @app.get("/termsofuse", response_class=HTMLResponse, tags=["Pages"])
    async def terms_of_use():
        return _read_html("termsofuse.html")

    @app.get("/privacy", response_class=HTMLResponse, tags=["Pages"])
    async def privacy_policy():
        return _read_html("privacy.html")
