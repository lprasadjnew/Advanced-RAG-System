# Advanced RAG System

A production-ready, domain-specific Retrieval-Augmented Generation system built with FastAPI, LangChain, Qdrant, and Google Gemini AI. Ingests documents, stores them in a vector database, and answers questions with persistent conversation memory.

**Version:** 1.0.1 &nbsp;|&nbsp; **Author:** Lakshmiprasad Janjam &nbsp;|&nbsp; **License:** MIT

---

## Features

- **Domain-Aware Ingestion** — Validates documents against a configured domain (e.g., health/wellness) and rejects off-topic files automatically
- **Content Deduplication** — SHA-256 content hashing prevents re-processing the same document twice
- **Hierarchical Chunking** — Splits text along semantic boundaries (paragraphs → sections → pages) for coherent retrieval units
- **Reranking** — Cross-encoder model (`ms-marco-MiniLM-L-6-v2`) reranks candidates for precision before generation
- **Conversation Memory** — Per-session chat history persisted in SQLite
- **Multi-Format Support** — PDF, DOCX, TXT, and Markdown documents
- **Dual Web UIs** — Separate Gradio interfaces for chatting and uploading documents
- **Docker Support** — Multi-stage Dockerfile and Docker Compose with Qdrant bundled
- **Health Endpoint** — `/health` returns live system status (LLM, Qdrant, collection)

---

## Project Structure

```
Advanced-RAG-System/
├── pyproject.toml                    # Project metadata and dependencies (PEP 517/518)
├── README.md
├── LICENSE                           # MIT License
├── LICENSES/
│   └── THIRD_PARTY_LICENSES.md      # Licenses for all third-party packages
├── .env.example                      # Configuration template
├── .gitignore
│
└── src/
    └── Advanced-RAG/
        ├── __init__.py
        ├── main.py                   # FastAPI application entry point
        ├── config.py                 # Pydantic Settings — all config from environment
        ├── database.py               # SQLAlchemy models & async SQLite session
        ├── routes_pages.py           # Static HTML page routes
        ├── requirements.txt          # Pinned dependencies (used by Docker build)
        ├── Dockerfile                # Multi-stage Docker image
        ├── docker-compose.yml        # App + Qdrant orchestration
        │
        ├── services/
        │   ├── rag_service.py        # Orchestrates ingestion and query pipelines
        │   ├── document_processor.py # Loads and chunks PDF/DOCX/TXT/MD files
        │   ├── vector_store.py       # Qdrant operations (upsert, retrieve, dedup)
        │   ├── domain_validator.py   # Gemini-based domain classification
        │   └── reranker.py           # Cross-encoder reranking
        │
        ├── ui/
        │   ├── chat_ui.py            # Gradio chat interface
        │   └── upload_ui.py          # Gradio document upload interface
        │
        └── static/
            ├── index.html            # Landing page
            ├── termsofuse.html
            └── privacy.html
```

### Pipeline Overview

**Ingestion**
```
Upload file → Domain validation (Gemini) → Load & chunk → Hash dedup check
    → Embed chunks (gemini-embedding-001) → Store in Qdrant
```

**Query**
```
User question → Embed query → Retrieve top-K chunks from Qdrant
    → Rerank with cross-encoder → Generate answer (Gemini) with conversation history
```

---

## Prerequisites

- Python 3.10+
- [Qdrant](https://qdrant.tech/) running (locally or via Docker)
- A [Google AI Studio](https://aistudio.google.com/) API key with access to Gemini

---

## Installation

### Option 1 — Local (virtual environment)

```bash
# 1. Clone the repository
git clone https://github.com/lprasadjnew/Advanced-RAG-System.git
cd Advanced-RAG-System

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install the package and dependencies
pip install -e .

# 4. Configure environment variables
cp .env.example src/Advanced-RAG/.env
# Edit .env and set your GOOGLE_API_KEY
```

Start Qdrant separately:

```bash
docker run -d -p 6333:6333 qdrant/qdrant
```

Run the app:

```bash
cd src/Advanced-RAG
python main.py
```

---

### Option 2 — Docker Compose (recommended for production)

```bash
# 1. Clone the repository
git clone https://github.com/lprasadjnew/Advanced-RAG-System.git
cd Advanced-RAG-System/src/Advanced-RAG

# 2. Configure environment variables
cp ../../.env.example .env
# Edit .env and set your GOOGLE_API_KEY

# 3. Build and start all services (app + Qdrant)
docker compose up --build -d
```

The app and Qdrant start together. Qdrant is reachable internally as `qdrant-service`.

Stop all services:

```bash
docker compose down
```

---

## Configuration

All configuration is read from the `.env` file (located at `src/Advanced-RAG/.env`). Copy `.env.example` as a starting point.

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | *(required)* | Google Gemini API key |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model used for generation |
| `GEMINI_EMBEDDING_MODEL` | `models/gemini-embedding-001` | Gemini model used for embeddings |
| `DOMAIN` | `health, nutrition, medicine, wellness, diet, fitness` | Comma-separated keywords defining the accepted domain |
| `DOMAIN_DESCRIPTION` | `Health and wellness knowledge base` | Human-readable domain label shown in UI |
| `QDRANT_HOST` | `localhost` | Qdrant host (`qdrant` when using Docker Compose) |
| `QDRANT_PORT` | `6333` | Qdrant port |
| `QDRANT_COLLECTION` | `rag_documents` | Qdrant collection name |
| `CHUNK_SIZE` | `1000` | Target chunk size in characters |
| `CHUNK_OVERLAP` | `200` | Overlap between consecutive chunks |
| `TOP_K_RETRIEVAL` | `10` | Candidates retrieved from Qdrant before reranking |
| `TOP_K_RERANKED` | `4` | Top chunks passed to the LLM after reranking |
| `APP_HOST` | `0.0.0.0` | Bind host |
| `APP_PORT` | `8001` | Bind port |
| `HF_HOME` | `.cache/huggingface` | HuggingFace model cache directory |
| `SENTENCE_TRANSFORMERS_HOME` | `.cache/torch/sentence_transformers` | Sentence Transformers cache directory |
| `GRADIO_TEMP_DIR` | `.uploads` | Temporary directory for Gradio file uploads |

**Domain reconfiguration** — to switch from health/wellness to another domain (e.g., legal, finance), update `DOMAIN` and `DOMAIN_DESCRIPTION` in `.env` and restart the app. No code changes needed.

---

## Usage

### 1. Upload Documents

Navigate to `http://localhost:8001/documentation`

- Drag and drop or select a **PDF, DOCX, TXT, or MD** file
- The system validates the file against the configured domain
- Off-topic documents are rejected with an explanation
- Accepted documents are chunked, embedded, and stored in Qdrant
- Duplicate documents (same content) are detected and skipped

### 2. Ask Questions

Navigate to `http://localhost:8001/aiquery`

- Type a question in the chat interface
- The system retrieves relevant chunks, reranks them, and generates a grounded answer using Gemini
- Conversation history is maintained per browser session in SQLite

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Landing page |
| `GET` | `/health` | JSON system status (LLM, Qdrant, collection) |
| `GET` | `/aiquery` | Gradio chat UI |
| `GET` | `/documentation` | Gradio document upload UI |
| `GET` | `/termsofuse` | Terms of use |
| `GET` | `/privacy` | Privacy policy |
| `GET` | `/docs` | Swagger / OpenAPI documentation |
| `GET` | `/redoc` | ReDoc documentation |

### Health Check Response

```json
{
  "status": "ok",
  "domain": "Health and wellness knowledge base",
  "llm": "gemini-2.5-flash",
  "qdrant": "connected",
  "collection": "rag_documents"
}
```

---

## Technology Stack

| Layer | Technology |
|---|---|
| Web framework | FastAPI + Uvicorn |
| Web UI | Gradio |
| LLM & embeddings | Google Gemini (`gemini-2.5-flash`, `gemini-embedding-001`) |
| Vector database | Qdrant |
| RAG framework | LangChain |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (sentence-transformers) |
| Document parsing | PyPDF, pdfplumber, python-docx |
| Conversation storage | SQLite via SQLAlchemy + aiosqlite |
| Configuration | Pydantic Settings |
| Packaging | pyproject.toml (PEP 517/518, setuptools) |
| Containerisation | Docker (multi-stage) + Docker Compose |

---

## Development Notes

- Model caches are stored under `.cache/` inside the app directory (`HF_HOME`, `SENTENCE_TRANSFORMERS_HOME`) to keep the environment self-contained
- The cross-encoder model is downloaded on first startup (~90 MB)
- Qdrant collection is created automatically on first run if it does not exist
- The SQLite database (`conversations.db`) is created automatically on first run
- Uploaded files are temporarily staged during processing and not persisted

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

Third-party package licenses are listed in [LICENSES/THIRD_PARTY_LICENSES.md](LICENSES/THIRD_PARTY_LICENSES.md).
