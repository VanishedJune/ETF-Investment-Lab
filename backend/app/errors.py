"""Application errors whose public HTTP representation is fixed and safe."""

from __future__ import annotations


class PublicApiError(Exception):
    """Base for expected application failures safe to classify publicly."""

    status_code = 500
    public_detail = "Internal server error"

    def __init__(self, internal_message: str | None = None) -> None:
        super().__init__(internal_message or self.public_detail)
        self.internal_message = internal_message


class PublicValidationError(PublicApiError, ValueError):
    status_code = 400
    public_detail = "Invalid request"


class PublicNotFoundError(PublicApiError, ValueError):
    status_code = 404
    public_detail = "Resource not found"


class PublicConflictError(PublicApiError, ValueError):
    status_code = 409
    public_detail = "Request conflicts with current state"


class PublicDataUnavailableError(PublicApiError, ValueError):
    status_code = 503
    public_detail = "Data is temporarily unavailable"


__all__ = [
    "PublicApiError",
    "PublicConflictError",
    "PublicDataUnavailableError",
    "PublicNotFoundError",
    "PublicValidationError",
]
