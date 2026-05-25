import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from rag_agent.api.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
async def async_client() -> AsyncClient:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
