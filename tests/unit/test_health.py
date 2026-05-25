from fastapi.testclient import TestClient


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_metrics_endpoint(client: TestClient) -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests_total" in response.text
