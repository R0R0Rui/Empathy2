"""
schemas.py
==========
The output contract for the LLM #1 Understanding layer.

Anything that imports from here is depending on these field names.
Treat them as the stable handoff API to:
  - app.main (which adds `understanding` to /chat response)
  - the LLM #2 prompt builder (uses support_need + intent + tier)
  - the future evaluation phase (uses scenario_tier as the eval label)

Design notes:
  - We use Pydantic v2 because the rest of the FastAPI backend already does.
    No new dependencies added.
  - We keep GoEmotions labels in primary_emotion / secondary_emotions so the
    output is consistent with the team's existing /chat 'emotions' field.
  - personality is OPTIONAL and only emitted once confidence clears the
    floor (see personality_tracker.py). Default is None.
"""

from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Allowed enums (also re-exported in __init__ so external code can import)
# ---------------------------------------------------------------------------
ALLOWED_INTENTS = (
    "emotional_support",
    "advice_request",
    "venting",
    "reflection",
    "information_request",
    "planning_help",
    "relationship_support",
    "work_or_school_stress",
    "health_or_body_concern",
    "crisis_or_safety",
    "small_talk",
    "other",
)

ALLOWED_SUPPORT_NEEDS = (
    "validation",
    "validation_and_gentle_exploration",
    "gentle_exploration",
    "problem_solving",
    "grounding",
    "encouragement",
    "normalization_without_dismissal",
    "safety_support",
    "resource_referral",
    "clarifying_question",
)

ALLOWED_TIERS = ("common", "subtle", "high_risk")
ALLOWED_SAFETY_FLAGS = ("none", "low", "medium", "high")


# ---------------------------------------------------------------------------
# Big Five (optional, multi-turn, low-confidence)
# ---------------------------------------------------------------------------
class BigFive(BaseModel):
    """OCEAN scores in [0, 1]. 0.5 means 'no signal'."""
    openness: float = Field(ge=0.0, le=1.0)
    conscientiousness: float = Field(ge=0.0, le=1.0)
    extraversion: float = Field(ge=0.0, le=1.0)
    agreeableness: float = Field(ge=0.0, le=1.0)
    neuroticism: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Main output: UnderstandingState
# ---------------------------------------------------------------------------
class UnderstandingState(BaseModel):
    """
    Final structured output from the Understanding layer.

    Field meanings:
      primary_emotion / secondary_emotions
          GoEmotions labels (matching app/models/emotion.py).
      emotion_intensity
          0..1 — how strongly the emotion is felt. Different from raw
          classifier confidence (which is the model's certainty).
      valence
          -1..+1 from NRC VAD lexicon. Negative = unpleasant.
      arousal
          0..1 from NRC VAD lexicon. 0 = calm, 1 = activated.
      intent
          What the user *wants* (not what they feel).
      is_implicit
          True if the user did NOT explicitly name a feeling in their text.
          Drives the 'subtle' scenario tier.
      scenario_tier
          common | subtle | high_risk — matches the proposal eval slide.
      safety_flag
          none | low | medium | high — used by main.py to short-circuit
          to the existing crisis_reply() if 'high'.
      crisis_signals
          List of human-readable strings explaining safety_flag.
      support_need
          A directive for LLM #2 / response generator: what kind of support
          to give. Picked from ALLOWED_SUPPORT_NEEDS.
      evidence
          1-3 short phrases explaining the decision. Helps the team debug
          and helps the future evaluation step audit results.
      personality
          Optional Big-Five snapshot, only present when confidence is
          high enough across multi-turn observation.
      personality_confidence
          0..1 — grows with more observed user text.
      pii_redacted_text
          The user's text with email/phone/URL replaced. Use for logging
          and screenshots, not for the LLM call itself.
      layer_outputs
          Raw outputs from each internal layer, for transparency / eval.
          Treat as debug-only — don't depend on its shape.
    """

    primary_emotion: str
    secondary_emotions: List[str] = Field(default_factory=list)

    emotion_intensity: float = Field(ge=0.0, le=1.0)
    valence: float = Field(ge=-1.0, le=1.0)
    arousal: float = Field(ge=0.0, le=1.0)

    intent: str
    is_implicit: bool

    scenario_tier: Literal["common", "subtle", "high_risk"]
    safety_flag: Literal["none", "low", "medium", "high"] = "none"
    crisis_signals: List[str] = Field(default_factory=list)

    support_need: str
    evidence: List[str] = Field(default_factory=list)

    personality: Optional[BigFive] = None
    personality_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    pii_redacted_text: str

    layer_outputs: Dict[str, Any] = Field(default_factory=dict)
