"""
RAG System — FastAPI entry point.

Endpoints:
  GET  /aiquery        → Gradio chat interface
  GET  /documentation  → Gradio document upload interface
  GET  /health         → JSON health-check
"""

import os
import pathlib
from routes_pages import register_page_routes

# Tell Gradio to stage uploaded files inside our project directory
# instead of /tmp/gradio (which may be owned by another user/process).
_UPLOAD_DIR = pathlib.Path(__file__).parent / "uploads"
_UPLOAD_DIR.mkdir(exist_ok=True)
os.environ.setdefault("GRADIO_TEMP_DIR", str(_UPLOAD_DIR))

from contextlib import asynccontextmanager

import gradio as gr
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import settings
from database import init_db
from services.vector_store import ensure_collection
from ui.chat_ui import create_chat_interface, CHAT_THEME, CHAT_CSS
from ui.upload_ui import create_upload_interface, UPLOAD_THEME, UPLOAD_CSS

# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    ensure_collection()
    print(f"[startup] Domain        : {settings.DOMAIN_DESCRIPTION}")
    print(f"[startup] Qdrant        : {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
    print(f"[startup] Collection    : {settings.QDRANT_COLLECTION}")
    print(f"[startup] LLM model     : {settings.GEMINI_MODEL}")
    print(f"[startup] Chat UI       : http://localhost:{settings.APP_PORT}/aiquery")
    print(f"[startup] Upload UI     : http://localhost:{settings.APP_PORT}/documentation")
    yield


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Domain RAG System",
    description=f"RAG system for: {settings.DOMAIN_DESCRIPTION}",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

register_page_routes(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health() -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "domain": settings.DOMAIN_DESCRIPTION,
        "llm": settings.GEMINI_MODEL,
        "qdrant": f"{settings.QDRANT_HOST}:{settings.QDRANT_PORT}",
        "collection": settings.QDRANT_COLLECTION,
    })


# ── Mount Gradio interfaces ───────────────────────────────────────────────────

chat_blocks = create_chat_interface()
upload_blocks = create_upload_interface()

app = gr.mount_gradio_app(
    app, chat_blocks,
    path="/aiquery",
    theme=CHAT_THEME,
    css=CHAT_CSS,
)
app = gr.mount_gradio_app(
    app, upload_blocks,
    path="/documentation",
    theme=UPLOAD_THEME,
    css=UPLOAD_CSS,
    max_file_size="5mb",
)


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=False,
    )
