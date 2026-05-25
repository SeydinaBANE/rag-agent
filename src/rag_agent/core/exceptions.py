from fastapi import HTTPException, status


class RagAgentError(Exception):
    """Base exception for all domain errors."""

    def __init__(self, message: str, code: str = "INTERNAL_ERROR") -> None:
        super().__init__(message)
        self.message = message
        self.code = code


class DocumentNotFoundError(RagAgentError):
    def __init__(self, doc_id: str) -> None:
        super().__init__(f"Document '{doc_id}' not found", code="DOCUMENT_NOT_FOUND")


class IngestError(RagAgentError):
    def __init__(self, reason: str) -> None:
        super().__init__(f"Ingestion failed: {reason}", code="INGEST_ERROR")


class LLMError(RagAgentError):
    def __init__(self, reason: str) -> None:
        super().__init__(f"LLM call failed: {reason}", code="LLM_ERROR")


class GuardrailError(RagAgentError):
    def __init__(self, reason: str) -> None:
        super().__init__(f"Guardrail blocked request: {reason}", code="GUARDRAIL_BLOCKED")


class AuthError(HTTPException):
    def __init__(self, detail: str = "Invalid or missing API key") -> None:
        super().__init__(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail)


class RateLimitError(HTTPException):
    def __init__(self) -> None:
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Retry after 60 seconds.",
        )
