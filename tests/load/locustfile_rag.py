from locust import HttpUser, between, task


class RagUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def chat(self) -> None:
        self.client.post(
            "/api/v1/chat",
            json={"query": "What is RAG?", "session_id": "load-test"},
            headers={"X-API-Key": "test-key"},
        )

    @task(1)
    def health(self) -> None:
        self.client.get("/health")
