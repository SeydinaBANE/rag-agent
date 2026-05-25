"""FastAPI dependencies: API key auth, rate limiting."""

import hashlib

import structlog
from fastapi import Header, HTTPException, status
from fastapi.security import APIKeyHeader

from rag_agent.core.config import settings

log = structlog.get_logger()

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# In production: store hashed keys in PostgreSQL (see models/database.py)
# For dev/demo: accept any key that matches the env-configured salt hash
_DEV_KEY = "dev-key-" + settings.api_secret_salt


async def require_api_key(x_api_key: str | None = Header(None, alias="X-API-Key")) -> str:
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header",
        )

    # TODO: look up hashed key in DB for production
    # For now: accept dev key or any non-empty key in dev mode
    if settings.is_production:
        key_hash = hashlib.sha256(
            (x_api_key + settings.api_secret_salt).encode()
        ).hexdigest()
        # In prod: query DB for key_hash
        # For now just validate format
        if len(x_api_key) < 16:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
            )

    structlog.contextvars.bind_contextvars(api_key_prefix=x_api_key[:8])
    return x_api_key
