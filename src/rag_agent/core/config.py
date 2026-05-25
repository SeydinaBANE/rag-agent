from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    openrouter_api_key: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    default_model: str = "google/gemini-flash-1.5"
    quality_model: str = "anthropic/claude-3.5-sonnet"

    # Database
    database_url: str

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # ChromaDB
    chroma_host: str = "localhost"
    chroma_port: int = 8001

    # MinIO
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "rag-documents"
    minio_secure: bool = False

    # Langfuse
    langfuse_secret_key: str = ""
    langfuse_public_key: str = ""
    langfuse_host: str = "http://localhost:3000"

    # Auth
    api_secret_salt: str = "changeme"

    # Rate limiting
    rate_limit_per_minute: int = 60

    # Guardrails
    guardrails_pii_enabled: bool = True
    guardrails_hallucination_threshold: float = 0.75
    guardrails_toxicity_enabled: bool = True

    # Semantic cache
    semantic_cache_enabled: bool = True
    semantic_cache_similarity_threshold: float = 0.92
    semantic_cache_ttl_seconds: int = 3600

    # RAG
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5

    # App
    app_env: str = "development"
    log_level: str = "INFO"

    @field_validator("app_env")
    @classmethod
    def validate_env(cls, v: str) -> str:
        if v not in ("development", "production"):
            raise ValueError("app_env must be 'development' or 'production'")
        return v

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def chroma_url(self) -> str:
        return f"http://{self.chroma_host}:{self.chroma_port}"


settings = Settings()  # type: ignore[call-arg]
