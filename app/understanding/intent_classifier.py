"""
intent_classifier.py
====================
Maps a user message + recent history to a single 'intent' label.

WHY INTENT MATTERS (the proposal calls it "Emotion + Intent Understanding"):
  Two users can express the same emotion but want very different things:
    - "I'm so anxious about tomorrow."          -> venting (just heard out)
    - "I'm so anxious about tomorrow. What should I do?" -> advice_request
    - "I keep thinking everyone hates me."      -> seeking_validation
    - "I want to die."                          -> crisis_or_safety
  Without intent, LLM #2 picks the wrong response strategy.

DESIGN:
  Rule-based classifier using surface cues (questions, modal verbs, common
  phrases). Deterministic, no API call needed. Fast enough to run inline
  with /chat.

  This is INTENTIONALLY simple. We can plug in an LLM call later via
  llm_analyzer.py for cases where the rules disagree with each other.
"""

from __future__ import annotations
import re
from typing import List

from .schemas import ALLOWED_INTENTS


# Pre-compiled cue patterns. The first match wins, in priority order.
# Crisis cues are checked separately (they're handled by safety_fusion); here
# we just need 'crisis_or_safety' to be available as a valid output.
_ADVICE_CUES = [
    r"\bwhat should i\b",
    r"\bwhat do i do\b",
    r"\bhow do i\b",
    r"\bhow can i\b",
    r"\bany advice\b",
    r"\bany ideas?\b",
    r"\bany suggestions?\b",
    r"\bplease help\b",
    r"\bcan you help\b",
    r"\bdo you have any\b",
    # NOTE: a bare trailing "?" is intentionally NOT an advice cue here.
    # Many small-talk and rhetorical questions end in "?" without asking
    # for advice ("how are you?", "is that bad?"). We rely on the
    # specific patterns above instead.
]

_PLANNING_CUES = [
    r"\bplan\b", r"\bschedule\b", r"\borganize\b",
    r"\bnext steps?\b", r"\bstep[-\s]?by[-\s]?step\b",
]

_VALIDATION_CUES = [
    r"\bam i (?:wrong|right|crazy|overreacting|being)",
    r"\bis it ok\b", r"\bis it okay\b",
    r"\bdo you think\b",
    r"\bmaybe i\b",
]

_VENTING_CUES = [
    r"\bjust need(?:ed)? to (?:say|vent|talk)\b",
    r"\bjust venting\b",
    r"\bdon'?t need (?:advice|help)\b",
]

_INFO_CUES = [
    r"\bwhat is\b", r"\bwhat are\b",
    r"\bdefine\b", r"\bexplain\b",
    r"\btell me about\b",
]

_RELATIONSHIP_CUES = [
    r"\b(?:my )?(?:friend|partner|boyfriend|girlfriend|spouse|wife|husband|mom|mother|dad|father|brother|sister|family)\b",
]

_WORK_SCHOOL_CUES = [
    r"\b(?:work|job|boss|coworker|deadline|exam|class|professor|homework|school|college|university)\b",
]

_HEALTH_CUES = [
    r"\b(?:sleep|insomnia|sick|doctor|therapist|medication|pain|appetite|eating|exercise)\b",
]

_SMALL_TALK_CUES = [
    r"^\s*(?:hi|hello|hey|sup|yo)\b",
    r"\bhow are you\b",
]


def _any_match(text: str, patterns: List[str]) -> bool:
    return any(re.search(p, text) for p in patterns)


def classify_intent(
    user_text: str,
    primary_emotion: str,
    crisis_detected: bool,
) -> str:
    """
    Pick a single intent label from ALLOWED_INTENTS.

    Priority order:
      crisis > advice_request > planning > validation > venting >
      info > relationship > work/school > health > small_talk > emotional_support
    """
    if crisis_detected:
        return "crisis_or_safety"

    text_lc = user_text.lower()

    if _any_match(text_lc, _ADVICE_CUES):
        return "advice_request"
    if _any_match(text_lc, _PLANNING_CUES):
        return "planning_help"
    if _any_match(text_lc, _VALIDATION_CUES):
        return "reflection"
    if _any_match(text_lc, _VENTING_CUES):
        return "venting"
    if _any_match(text_lc, _INFO_CUES) and primary_emotion in {"neutral", "curiosity"}:
        return "information_request"

    # Topic cues — only if we already think there's emotional content
    has_emotion = primary_emotion not in {"neutral"}
    if has_emotion:
        if _any_match(text_lc, _RELATIONSHIP_CUES):
            return "relationship_support"
        if _any_match(text_lc, _WORK_SCHOOL_CUES):
            return "work_or_school_stress"
        if _any_match(text_lc, _HEALTH_CUES):
            return "health_or_body_concern"

    if _any_match(text_lc, _SMALL_TALK_CUES) and not has_emotion:
        return "small_talk"

    # Default: if there's an emotion, the user likely wants emotional_support
    return "emotional_support" if has_emotion else "other"


# Sanity guard so callers can't accidentally send an unknown label downstream
def normalize_intent(label: str) -> str:
    return label if label in ALLOWED_INTENTS else "other"
