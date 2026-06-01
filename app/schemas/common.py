"""Cross-cutting response schemas (errors, pagination)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """Uniform error envelope for non-2xx responses."""

    detail: str = Field(..., description="Human-readable error message")
    code: str | None = Field(None, description="Machine-readable error code")


class Pagination(BaseModel):
    """Cursor-less pagination metadata."""

    total: int = Field(..., ge=0, description="Total items across all pages")
    page: int = Field(..., ge=1, description="1-indexed page number")
    page_size: int = Field(..., ge=1, le=500, description="Items per page")

    @property
    def n_pages(self) -> int:
        return (self.total + self.page_size - 1) // max(self.page_size, 1)


class Page[T](BaseModel):
    """Generic paginated response (PEP 695 type parameter syntax)."""

    items: list[T]
    pagination: Pagination
