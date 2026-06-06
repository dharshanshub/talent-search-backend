from __future__ import annotations


class AppException(Exception):
    """Base exception for all application-level errors."""

    status_code: int = 500
    error_code: str = "INTERNAL_ERROR"

    def __init__(self, message: str = "An unexpected error occurred") -> None:
        super().__init__(message)
        self.message = message


class NotFoundError(AppException):
    status_code = 404
    error_code = "NOT_FOUND"

    def __init__(self, message: str = "Resource not found") -> None:
        super().__init__(message)


class BadRequestError(AppException):
    status_code = 400
    error_code = "BAD_REQUEST"

    def __init__(self, message: str = "Invalid request") -> None:
        super().__init__(message)


class UpstreamServiceError(AppException):
    """Raised when OpenAI or Pinecone returns an unexpected error."""

    status_code = 502
    error_code = "UPSTREAM_ERROR"

    def __init__(self, service: str, message: str = "Upstream service error") -> None:
        super().__init__(f"{service}: {message}")
        self.service = service


class ConfigurationError(AppException):
    status_code = 500
    error_code = "CONFIGURATION_ERROR"

    def __init__(self, message: str = "Server configuration error") -> None:
        super().__init__(message)
