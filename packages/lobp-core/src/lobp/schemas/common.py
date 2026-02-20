"""Common Pydantic schemas used across the application."""

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict

T = TypeVar("T")


class Message(BaseModel):
    """Simple message response."""

    message: str


class PaginatedResponse(BaseModel, Generic[T]):
    """Paginated response wrapper."""

    model_config = ConfigDict(from_attributes=True)

    items: list[T]
    total: int
    page: int
    page_size: int
    pages: int


class HealthCheck(BaseModel):
    """Health check response."""

    status: str
    version: str
    database: str
    redis: str


class ErrorResponse(BaseModel):
    """Error response schema."""

    detail: str
    code: str | None = None
    params: dict[str, Any] | None = None
