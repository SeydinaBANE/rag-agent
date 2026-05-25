"""FastAPI dependencies: API key auth."""

from __future__ import annotations

import hashlib

import structlog
from fastapi import Depends, Header, HTTPException, status
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from rag_agent.core.config import settings
from rag_agent.models.database import ApiKey, get_db

log = structlog.get_logger()

_DEV_KEY = "dev-key-" + settings.api_secret_salt


def hash_key(raw_key: str) -> str:
    """SHA-256(key + salt). Never store raw keys."""
    return hashlib.sha256((raw_key + settings.api_secret_salt).encode()).hexdigest()


async def require_api_key(
    x_api_key: str | None = Header(None, alias="X-API-Key"),
    db: AsyncSession = Depends(get_db),
) -> str:
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header",
        )

    # Dev shortcut — skip DB when using the well-known dev key
    if not settings.is_production and x_api_key == _DEV_KEY:
        structlog.contextvars.bind_contextvars(api_key_prefix=x_api_key[:8])
        return x_api_key

    row = (
        await db.execute(
            select(ApiKey).where(ApiKey.key_hash == hash_key(x_api_key), ApiKey.is_active.is_(True))
        )
    ).scalar_one_or_none()

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or revoked API key",
        )

    await db.execute(update(ApiKey).where(ApiKey.id == row.id).values(last_used_at=func.now()))
    await db.commit()

    structlog.contextvars.bind_contextvars(api_key_prefix=x_api_key[:8], key_id=str(row.id))
    return x_api_key
