"""Pydantic request/response models for the API."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000, description="The customer's question")
    max_new_tokens: int = Field(128, ge=1, le=512)
    temperature: float = Field(0.2, ge=0.0, le=2.0)


class GenerateResponse(BaseModel):
    response: str
    model_version: str
    prompt_version: str
    latency_ms: float
    tokens_generated: int
    request_id: str


class FeedbackRequest(BaseModel):
    request_id: str = Field(..., min_length=1, max_length=64)
    rating: Literal["up", "down"]
    comment: Optional[str] = Field(None, max_length=1000)


class FeedbackResponse(BaseModel):
    status: str = "recorded"
    request_id: str
