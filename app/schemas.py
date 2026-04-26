from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str | None = None
    history_limit: int | None = Field(default=None, ge=1, le=100)


class EmotionItem(BaseModel):
    label: str
    confidence: float


class EmotionPayload(BaseModel):
    primary: EmotionItem
    secondary: EmotionItem
    tertiary: EmotionItem


class SafetyPayload(BaseModel):
    crisis_detected: bool = False


class PerformancePayload(BaseModel):
    classifier_ms: float
    generation_ms: float
    total_ms: float


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    emotions: EmotionPayload
    performance: PerformancePayload
    safety: SafetyPayload = Field(default_factory=SafetyPayload)


class ResetResponse(BaseModel):
    session_id: str | None = None
    cleared_sessions: int | None = None
    ok: bool = True
