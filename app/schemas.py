from __future__ import annotations

from typing import Any

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
    # Optional: time spent in the Understanding layer (LLM #1 fusion).
    # Defaults to 0.0 if the analyzer is disabled.
    understanding_ms: float = 0.0
    control_ms: float = 0.0


# ---------------------------------------------------------------------------
# Understanding (LLM #1) layer — additive, optional
# ---------------------------------------------------------------------------
# These mirror app.understanding.schemas.UnderstandingState but live here
# so the API contract stays in one place and FastAPI / OpenAPI can render
# the schema without importing the internal module.

class BigFivePayload(BaseModel):
    openness: float = Field(ge=0.0, le=1.0)
    conscientiousness: float = Field(ge=0.0, le=1.0)
    extraversion: float = Field(ge=0.0, le=1.0)
    agreeableness: float = Field(ge=0.0, le=1.0)
    neuroticism: float = Field(ge=0.0, le=1.0)


class UnderstandingPayload(BaseModel):
    primary_emotion: str
    secondary_emotions: list[str] = Field(default_factory=list)
    emotion_intensity: float = Field(ge=0.0, le=1.0)
    valence: float = Field(ge=-1.0, le=1.0)
    arousal: float = Field(ge=0.0, le=1.0)
    intent: str
    is_implicit: bool
    scenario_tier: str
    safety_flag: str
    crisis_signals: list[str] = Field(default_factory=list)
    support_need: str
    evidence: list[str] = Field(default_factory=list)
    personality: BigFivePayload | None = None
    personality_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    pii_redacted_text: str
    layer_outputs: dict[str, Any] = Field(default_factory=dict)


class SupportPlanPayload(BaseModel):
    support_goal: str
    response_acts: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)
    safety_notes: list[str] = Field(default_factory=list)
    repair_priorities: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class ValidationPayload(BaseModel):
    passed: bool
    failure_types: list[str] = Field(default_factory=list)
    severity: str
    rewrite_needed: bool
    notes: list[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    anchor_reply: str | None = None
    raw_reply: str | None = None
    emotions: EmotionPayload
    performance: PerformancePayload
    safety: SafetyPayload = Field(default_factory=SafetyPayload)
    # Additive: present iff the Understanding analyzer ran successfully.
    # Existing clients that ignore unknown fields keep working.
    understanding: UnderstandingPayload | None = None
    support_plan: SupportPlanPayload | None = None
    validation: ValidationPayload | None = None
    refined_validation: ValidationPayload | None = None
    refined: bool = False
    refinement_reason: str | None = None
    failure_types: list[str] = Field(default_factory=list)


class ResetResponse(BaseModel):
    session_id: str | None = None
    cleared_sessions: int | None = None
    ok: bool = True
