"""
app.understanding
=================
LLM #1 Emotion + Intent Understanding layer for the Anchor Empathy AI agent.

Public surface (what other code imports):
    from app.understanding import UnderstandingAnalyzer, UnderstandingState

This module sits BETWEEN the existing GoEmotions classifier and the
response generator. It does NOT replace either; it produces a structured
"understanding" object that:
  - The LLM #2 generator uses to choose response strategy.
  - The future evaluation phase can use as ground truth for the
    Common/Subtle/High-risk scenario tiers shown in the proposal deck.
"""

from .schemas import (
    UnderstandingState,
    BigFive,
    ALLOWED_INTENTS,
    ALLOWED_SUPPORT_NEEDS,
    ALLOWED_TIERS,
    ALLOWED_SAFETY_FLAGS,
)
from .analyzer import UnderstandingAnalyzer

__all__ = [
    "UnderstandingAnalyzer",
    "UnderstandingState",
    "BigFive",
    "ALLOWED_INTENTS",
    "ALLOWED_SUPPORT_NEEDS",
    "ALLOWED_TIERS",
    "ALLOWED_SAFETY_FLAGS",
]
