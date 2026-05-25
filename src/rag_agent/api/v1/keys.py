"""API key management: create, list, revoke."""

from __future__ import annotations

import secrets
import uuid

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_agent.api.v1.deps import hash_key, require_api_key
from rag_agent.models.database import ApiKey, get_db
from rag_agent.models.schemas import KeyCreate, KeyCreated, KeyInfo

log = structlog.get_logger()
router = APIRouter(prefix="/keys", tags=["keys"])


@router.post("", response_model=KeyCreated, status_code=status.HTTP_201_CREATED)
async def create_key(
    body: KeyCreate,
    db: AsyncSession = Depends(get_db),
    _: str = Depends(require_api_key),
) -> KeyCreated:
    """Create a new API key. The plaintext key is returned exactly once — store it securely."""
    raw_key = secrets.token_urlsafe(32)
    row = ApiKey(id=uuid.uuid4(), key_hash=hash_key(raw_key), name=body.name)
    db.add(row)
    await db.commit()
    await db.refresh(row)
    log.info("api_key_created", name=body.name, key_id=str(row.id))
    return KeyCreated(id=row.id, name=row.name, key=raw_key, created_at=row.created_at)


@router.get("", response_model=list[KeyInfo])
async def list_keys(
    db: AsyncSession = Depends(get_db),
    _: str = Depends(require_api_key),
) -> list[KeyInfo]:
    rows = (await db.execute(select(ApiKey).where(ApiKey.is_active.is_(True)))).scalars().all()
    return [KeyInfo(id=r.id, name=r.name, created_at=r.created_at, last_used_at=r.last_used_at) for r in rows]


@router.delete("/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_key(
    key_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    _: str = Depends(require_api_key),
) -> None:
    row = (await db.execute(select(ApiKey).where(ApiKey.id == key_id))).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Key not found")
    row.is_active = False
    await db.commit()
    log.info("api_key_revoked", key_id=str(key_id))
