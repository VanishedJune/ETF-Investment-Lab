"""Package-level ASGI entrypoint kept for conventional FastAPI deployment."""

from backend.web import app

__all__ = ["app"]
