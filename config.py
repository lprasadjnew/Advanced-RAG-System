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
    QDRANT_HOST: str = "localhost"
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

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
