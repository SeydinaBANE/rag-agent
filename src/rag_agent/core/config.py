from pydantic import Field, field_validator, model_validator
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
    rate_limit_per_minute: int = Field(default=60, ge=1, le=10000)

    # Guardrails
    guardrails_pii_enabled: bool = True
    guardrails_hallucination_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    guardrails_toxicity_enabled: bool = True

    # Semantic cache
    semantic_cache_enabled: bool = True
    semantic_cache_similarity_threshold: float = Field(default=0.92, ge=0.0, le=1.0)
    semantic_cache_ttl_seconds: int = Field(default=3600, ge=1)

    # RAG
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Agent — tunable without Docker rebuild
    max_agent_steps: int = Field(default=8, ge=1, le=50)
    agent_context_steps: int = Field(default=6, ge=1, le=20)
    agent_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    agent_max_tokens: int = Field(default=600, ge=64, le=8192)
    max_observation_length: int = Field(default=1500, ge=100)
    max_retrieval_retries: int = Field(default=2, ge=0, le=10)
    hallucination_check_chunks: int = Field(default=3, ge=1, le=20)
    # Threshold for triggering web-search fallback during retrieval grading
    # (distinct from guardrails_hallucination_threshold which governs post-generation retry)
    web_search_fallback_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    # Agent tools
    web_search_results: int = Field(default=5, ge=1, le=20)
    fetch_url_max_chars: int = Field(default=3000, ge=100)
    sql_max_rows: int = Field(default=50, ge=1, le=1000)

    # OpenTelemetry — set to OTLP endpoint (e.g. http://jaeger:4317) to export traces
    otel_exporter_endpoint: str = ""

    # App
    app_env: str = "development"
    log_level: str = "INFO"

    @field_validator("app_env")
    @classmethod
    def validate_env(cls, v: str) -> str:
        if v not in ("development", "production"):
            raise ValueError("app_env must be 'development' or 'production'")
        return v

    @model_validator(mode="after")
    def validate_production_secrets(self) -> "Settings":
        if self.is_production and self.api_secret_salt == "changeme":  # pragma: allowlist secret
            raise ValueError("api_secret_salt must be changed from default in production")
        return self

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def chroma_url(self) -> str:
        return f"http://{self.chroma_host}:{self.chroma_port}"


settings = Settings()  # type: ignore[call-arg]
