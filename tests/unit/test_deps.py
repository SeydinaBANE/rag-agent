"""Tests for deps — API key auth dependency."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from rag_agent.api.v1.deps import hash_key, require_api_key


def test_hash_key_deterministic():
    """Same input always produces same hash."""
    h1 = hash_key("my-key")
    h2 = hash_key("my-key")
    assert h1 == h2


def test_hash_key_different_inputs():
    assert hash_key("key-a") != hash_key("key-b")


@pytest.mark.asyncio
async def test_require_api_key_missing_header():
    mock_db = AsyncMock()
    with pytest.raises(HTTPException) as exc_info:
        await require_api_key(x_api_key=None, db=mock_db)
    assert exc_info.value.status_code == 401
    assert "Missing" in exc_info.value.detail


@pytest.mark.asyncio
async def test_require_api_key_dev_shortcut():
    mock_db = AsyncMock()
    mock_settings = MagicMock()
    mock_settings.is_production = False

    with (
        patch("rag_agent.api.v1.deps.settings", mock_settings),
        patch("rag_agent.api.v1.deps._DEV_KEY", "dev-key-changeme"),
    ):
        result = await require_api_key(
            x_api_key="dev-key-changeme",  # pragma: allowlist secret
            db=mock_db,
        )

    assert result == "dev-key-changeme"


@pytest.mark.asyncio
async def test_require_api_key_valid_db_key():
    mock_row = MagicMock()
    mock_row.id = 1
    mock_db = AsyncMock()
    mock_scalar = MagicMock()
    mock_scalar.scalar_one_or_none = MagicMock(return_value=mock_row)
    mock_db.execute = AsyncMock(return_value=mock_scalar)
    mock_db.commit = AsyncMock()

    mock_settings = MagicMock()
    mock_settings.is_production = True

    with (
        patch("rag_agent.api.v1.deps.settings", mock_settings),
        patch("rag_agent.api.v1.deps.hash_key", return_value="hashed"),
    ):
        result = await require_api_key(
            x_api_key="valid-api-key",  # pragma: allowlist secret
            db=mock_db,
        )

    assert result == "valid-api-key"
    mock_db.commit.assert_called_once()


@pytest.mark.asyncio
async def test_require_api_key_invalid_db_key():
    mock_db = AsyncMock()
    mock_scalar = MagicMock()
    mock_scalar.scalar_one_or_none = MagicMock(return_value=None)
    mock_db.execute = AsyncMock(return_value=mock_scalar)

    mock_settings = MagicMock()
    mock_settings.is_production = True

    with (
        patch("rag_agent.api.v1.deps.settings", mock_settings),
        patch("rag_agent.api.v1.deps.hash_key", return_value="hashed"),
        pytest.raises(HTTPException) as exc_info,
    ):
        await require_api_key(x_api_key="bad-key", db=mock_db)  # pragma: allowlist secret

    assert exc_info.value.status_code == 401
    assert "Invalid" in exc_info.value.detail
