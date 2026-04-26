"""
implicit_detector.py
====================
Decide whether the user EXPLICITLY named an emotion or only IMPLIED it.

WHY THIS MATTERS:
  This is the single most important signal for the 'subtle' scenario tier
  in the proposal. Examples:
    explicit:  "I feel really sad today."         -> common
    explicit:  "I am exhausted and stressed."     -> common
    implicit:  "I just stayed in bed all day."    -> subtle
    implicit:  "Yeah... I guess. Whatever."       -> subtle
    implicit:  "It's fine. Everyone says I'm fine, so I must be."  -> subtle
    implicit:  "I should be happy. The project went well." -> subtle  (suppressed feeling)

  GoEmotions alone tends to mark implicit cases as 'neutral' with high
  confidence, completely missing the hidden feeling. By detecting
  implicitness here, we route those cases into the 'subtle' tier where
  LLM #2 can use a different response strategy.

DECISION RULE:
  - 'Suppressed feeling' patterns ("should be happy", "supposed to feel
    fine") OVERRIDE explicit-word matches, because they signal that the
    user is performing a feeling, not having one.
  - Otherwise, implicit if NONE of:
      - The user's text contains an explicit feeling word
      - The classifier returned a non-neutral label with confidence >= 0.45
"""

from __future__ import annotations
import re

from .lexical_features import LexicalFeatures


# Phrases that mean the user is *describing* a feeling they think they
# *should* have, not one they actually have. Strong signal that the real
# feeling is suppressed and the case should be treated as subtle.
_SUPPRESSED_PATTERNS = [
    r"\bi should (?:be|feel)\s+\w+",
    r"\bsupposed to (?:be|feel)\s+\w+",
    r"\bguess i should\b",
    r"\bmust be\s+(?:fine|happy|ok|okay)\b",
    r"\beveryone says i'?m (?:fine|happy|ok|okay)\b",
]


def _has_suppressed_feeling_pattern(text: str) -> bool:
    text_lc = text.lower()
    return any(re.search(p, text_lc) for p in _SUPPRESSED_PATTERNS)


def is_implicit_emotion(
    lex: LexicalFeatures,
    classifier_top_label: str,
    classifier_top_confidence: float,
    classifier_neutral_threshold: float = 0.45,
    raw_text: str = "",
) -> bool:
    """Return True if the user did NOT name a feeling explicitly."""
    # Suppressed-feeling override takes precedence
    if raw_text and _has_suppressed_feeling_pattern(raw_text):
        return True

    has_explicit_word = bool(lex.explicit_emotion_words)
    classifier_confident = (
        classifier_top_label not in {"neutral"}
        and classifier_top_confidence >= classifier_neutral_threshold
    )
    return not (has_explicit_word or classifier_confident)
