import pytest
from fastapi import Header, HTTPException
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from rag_agent.api.main import app
from rag_agent.api.v1.deps import require_api_key


async def _bypass_auth(x_api_key: str | None = Header(None, alias="X-API-Key")) -> str:
    """Skip the DB lookup in tests — just check the header is present."""
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
    return x_api_key


app.dependency_overrides[require_api_key] = _bypass_auth


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
async def async_client() -> AsyncClient:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
