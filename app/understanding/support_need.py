"""
support_need.py
===============
Picks the right SUPPORT STRATEGY label for LLM #2 to follow.

THIS IS THE BRIDGE TO LLM #2:
  LLM #2's job is to GENERATE the empathetic reply. To do that well, it
  needs to know what kind of support is appropriate. Think of this as
  a "tone instruction" that gets injected into the response prompt.

EXAMPLES:
  user is venting               -> "validation"
  user asks for advice          -> "problem_solving"
  user implies sadness          -> "validation_and_gentle_exploration"
  high-risk safety              -> "safety_support"   (always wins)
  user is anxious               -> "grounding"
  user feels guilty/ashamed     -> "normalization_without_dismissal"

The LLM #2 prompt builder (app/prompting.py) can then translate these
labels into concrete instructions like "Acknowledge feelings, then ask
one open-ended question."
"""

from __future__ import annotations

from .schemas import ALLOWED_SUPPORT_NEEDS


def pick_support_need(
    primary_emotion: str,
    intent: str,
    safety_flag: str,
    is_implicit: bool,
) -> str:
    """Map (emotion, intent, safety, implicitness) -> support_need label."""
    if safety_flag == "high":
        return "safety_support"
    if safety_flag == "medium":
        return "validation_and_gentle_exploration"

    if intent in {"advice_request", "planning_help"}:
        return "problem_solving"

    if intent == "information_request":
        return "clarifying_question"

    if intent == "small_talk":
        return "validation"

    if primary_emotion in {"fear", "nervousness", "anxiety"} or primary_emotion == "stress":
        return "grounding"

    if primary_emotion in {"embarrassment", "remorse", "guilt", "shame"}:
        return "normalization_without_dismissal"

    if primary_emotion in {"sadness", "loneliness", "grief"} or is_implicit:
        return "validation_and_gentle_exploration"

    if primary_emotion in {"joy", "gratitude", "admiration", "optimism", "love", "excitement"}:
        return "encouragement"

    return "validation"


def normalize_support_need(label: str) -> str:
    return label if label in ALLOWED_SUPPORT_NEEDS else "validation"
