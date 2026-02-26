# NewRAG — Domain-Specific Retrieval-Augmented Generation System

A production-ready RAG system that ingests domain-specific documents, stores them in a vector database, and answers questions using Google Gemini with persistent conversation memory.

---

## Features

- **Domain-Aware Ingestion** — Validates documents against a configured domain (e.g., health/wellness) and rejects off-topic files automatically
- **Content Deduplication** — SHA-256 content hashing prevents re-processing the same document twice
- **Hierarchical Chunking** — Splits text along semantic boundaries (paragraphs → sections → pages) for coherent retrieval units
- **Reranking** — Cross-encoder model (`ms-marco-MiniLM-L-6-v2`) reranks candidates for precision before generation
- **Conversation Memory** — Per-session chat history persisted in SQLite
- **Multi-Format Support** — PDF, DOCX, TXT, and Markdown documents
- **Dual Web UIs** — Separate Gradio interfaces for chatting and uploading documents
- **Health Endpoint** — `/health` returns live system status (LLM, Qdrant, collection)

---

## Architecture

```
/var/www/NewRAG/
├── main.py                   # FastAPI application entry point
├── config.py                 # Pydantic Settings — all config from environment
├── database.py               # SQLAlchemy models & async SQLite session
├── routes_pages.py           # Static HTML page routes
├── requirements.txt
├── .env.example              # Configuration template
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
    → Embed chunks (text-embedding-004) → Store in Qdrant
```

**Query**
```
User question → Embed query → Retrieve top-K chunks from Qdrant
    → Rerank with cross-encoder → Generate answer (Gemini) with conversation history
```

---

## Prerequisites

- Python 3.12+
- [Qdrant](https://qdrant.tech/) running on `localhost:6333`
- A [Google AI Studio](https://aistudio.google.com/) API key with access to Gemini

---

## Installation

```bash
# 1. Clone / enter the project directory
cd /var/www/NewRAG

# 2. Create and activate a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env and set your GOOGLE_API_KEY (and any other values you want to override)
```

### Start Qdrant

Using Docker (simplest):

```bash
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

Or follow the [Qdrant installation guide](https://qdrant.tech/documentation/guides/installation/) for a standalone binary.

---

## Configuration

All configuration is read from the `.env` file. Copy `.env.example` and edit as needed.

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | *(required)* | Google Gemini API key |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model used for generation |
| `GEMINI_EMBEDDING_MODEL` | `models/text-embedding-004` | Gemini model used for embeddings |
| `DOMAIN` | `health, nutrition, medicine, wellness, diet, fitness` | Comma-separated keywords defining the accepted domain |
| `DOMAIN_DESCRIPTION` | `Health and wellness knowledge base` | Human-readable domain label (shown in UI) |
| `QDRANT_HOST` | `localhost` | Qdrant host |
| `QDRANT_PORT` | `6333` | Qdrant port |
| `QDRANT_COLLECTION` | `rag_documents` | Qdrant collection name |
| `CHUNK_SIZE` | `1000` | Target chunk size in characters |
| `CHUNK_OVERLAP` | `200` | Overlap between consecutive chunks |
| `TOP_K_RETRIEVAL` | `10` | Candidates retrieved from Qdrant before reranking |
| `TOP_K_RERANKED` | `4` | Top chunks passed to the LLM after reranking |
| `APP_HOST` | `0.0.0.0` | Bind host |
| `APP_PORT` | `8001` | Bind port |

**Domain reconfiguration** — to switch from health/wellness to another domain (e.g., legal, finance), update `DOMAIN` and `DOMAIN_DESCRIPTION` in `.env` and restart the app. No code changes are needed.

---

## Running

```bash
# Activate the virtual environment if not already active
source .venv/bin/activate

# Start the server
python main.py
```

Or with Uvicorn directly (development mode with auto-reload):

```bash
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

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
| Web UI | Gradio 5 |
| LLM & embeddings | Google Gemini (`gemini-2.5-flash`, `text-embedding-004`) |
| Vector database | Qdrant |
| RAG framework | LangChain |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (sentence-transformers) |
| Document parsing | PyPDF, pdfplumber, python-docx |
| Conversation storage | SQLite via SQLAlchemy + aiosqlite |
| Configuration | Pydantic Settings |

---

## Development Notes

- Model caches are stored under `.cache/` inside the project directory (`HF_HOME`, `SENTENCE_TRANSFORMERS_HOME`) to keep the environment self-contained
- The cross-encoder model is downloaded on first startup (~90 MB)
- Qdrant collection is created automatically on first run if it does not exist
- The SQLite database file (`conversations.db`) is created automatically on first run
- Uploaded files are temporarily staged in `uploads/` during processing and not persisted

---

## License

This project is provided as-is for internal/private use. See `static/termsofuse.html` for terms of use and `static/privacy.html` for the privacy policy.
