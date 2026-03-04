import os
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Google Gemini
    GOOGLE_API_KEY: str
    GEMINI_MODEL: str = "gemini-2.5-flash"
    GEMINI_EMBEDDING_MODEL: str = "models/text-embedding-004"

    # Domain
    DOMAIN: str = "health, nutrition, medicine, wellness, diet, fitness"
    DOMAIN_DESCRIPTION: str = "Health and wellness knowledge base"

    # Qdrant
    QDRANT_HOST: str 
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "rag_documents"

    # RAG
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RETRIEVAL: int = 10
    TOP_K_RERANKED: int = 4

    # App
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8001

    # Cache
    HF_HOME: str 
    SENTENCE_TRANSFORMERS_HOME: str 
    GRADIO_TEMP_DIR: str = ".uploads"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra="ignore"

settings = Settings()

# Project root (independent of working directory)
BASE_DIR = Path(__file__).resolve().parents[2]

def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return BASE_DIR / path

# Resolve paths

GRADIO_TEMP_DIR = resolve_path(settings.GRADIO_TEMP_DIR)
HF_HOME = resolve_path(settings.HF_HOME)
SENTENCE_TRANSFORMERS_HOME = resolve_path(settings.SENTENCE_TRANSFORMERS_HOME)

# Ensure directories exist

GRADIO_TEMP_DIR.mkdir(parents=True, exist_ok=True)
HF_HOME.mkdir(parents=True, exist_ok=True)
SENTENCE_TRANSFORMERS_HOME.mkdir(parents=True, exist_ok=True)

# Export to environment for libraries that expect env vars

os.environ["GRADIO_TEMP_DIR"] = str(GRADIO_TEMP_DIR)
os.environ["HF_HOME"] = str(HF_HOME)
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(SENTENCE_TRANSFORMERS_HOME)
