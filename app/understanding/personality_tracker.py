"""
personality_tracker.py
======================
Optional Big-Five (OCEAN) tendency tracker — multi-turn, low-confidence.


  ONLY as a low-confidence multi-turn tendency estimate, never as
  a diagnosis. Personality estimates from short text are notoriously noisy.
  Published LIWC personality work (Pennebaker & King 1999; Yarkoni 2010)
  used essays of ~2,500+ words; Twitter studies require 200+ tweets.

  Conclusion: it's defensible to expose a *tendency* signal but only after
  enough text has been observed. Below the threshold we expose `None`,
  which downstream treats as 'no signal'.

CONFIDENCE POLICY (intentionally conservative):
    user_words   |  confidence
    -----------------------
        0..150   |   0.0..0.2
       150..400  |   0.2..0.35
       400..800  |   0.35..0.5
       800..1500 |   0.5..0.65
       1500+     |   0.65 (capped — even 1500 words isn't enough for diagnosis)

  Below confidence_floor (default 0.4), we return `(None, confidence)` so
  the API never publishes a personality block prematurely.

PUBLIC INTERFACE:
    estimate_personality(history_user_text) -> (BigFive | None, confidence)
"""

from __future__ import annotations
import re
from typing import Dict, List, Optional, Tuple

from .schemas import BigFive


# ---------------------------------------------------------------------------
# Trait-correlated keyword sets
# Adapted from LIWC-Big-Five literature; intentionally small and inspectable.
# These are NOT a substitute for fine-tuned personality models; they're a
# starter signal so the system isn't silent on long conversations.
# ---------------------------------------------------------------------------
TRAIT_WORDS: Dict[str, Dict[str, List[str]]] = {
    "openness": {
        "positive": ["imagine", "dream", "creative", "wonder", "curious",
                     "explore", "idea", "perhaps", "maybe", "metaphor"],
        "negative": ["routine", "boring", "same", "avoid"],
    },
    "conscientiousness": {
        "positive": ["plan", "organize", "schedule", "responsible",
                     "deadline", "should", "must", "list", "prepare"],
        "negative": ["messy", "late", "forgot", "procrastinate", "chaotic"],
    },
    "extraversion": {
        "positive": ["we", "us", "friend", "friends", "party", "share",
                     "social", "group", "fun", "talk"],
        "negative": ["alone", "quiet", "isolated", "myself"],
    },
    "agreeableness": {
        "positive": ["thanks", "please", "kind", "appreciate", "caring",
                     "help", "sorry", "respect", "understand"],
        "negative": ["hate", "stupid", "blame", "rude", "annoying"],
    },
    "neuroticism": {
        "positive": ["anxious", "worried", "nervous", "scared", "panic",
                     "afraid", "depressed", "stressed", "hopeless", "lonely"],
        "negative": ["calm", "steady", "confident", "fine", "okay"],
    },
}

PERSONALITY_CONFIDENCE_FLOOR = 0.4


def _confidence_from_word_count(words: int) -> float:
    """Piecewise function described in the module docstring."""
    if words >= 1500:
        return 0.65
    if words >= 800:
        return 0.5 + (words - 800) / (1500 - 800) * (0.65 - 0.5)
    if words >= 400:
        return 0.35 + (words - 400) / (800 - 400) * (0.5 - 0.35)
    if words >= 150:
        return 0.2 + (words - 150) / (400 - 150) * (0.35 - 0.2)
    return min(0.2, words / 150 * 0.2)


def _trait_score(text_tokens: set, trait: str) -> float:
    """Score in [0, 1] with 0.5 = no signal."""
    pos_words = TRAIT_WORDS[trait]["positive"]
    neg_words = TRAIT_WORDS[trait]["negative"]
    pos_hits = sum(1 for w in pos_words if w in text_tokens)
    neg_hits = sum(1 for w in neg_words if w in text_tokens)
    raw = 0.5 + 0.05 * (pos_hits - neg_hits)
    return max(0.0, min(1.0, raw))


def estimate_personality(
    user_text_concat: str,
    confidence_floor: float = PERSONALITY_CONFIDENCE_FLOOR,
) -> Tuple[Optional[BigFive], float]:
    """
    Return (BigFive | None, confidence in [0,1]).

    `user_text_concat` is all of the user's text across the conversation,
    concatenated into one string. The session_store gives us this through
    history filtering for role == "user" and joining with newlines.
    """
    tokens = set(re.findall(r"[a-z']+", user_text_concat.lower()))
    word_count = len(re.findall(r"[a-z']+", user_text_concat.lower()))
    confidence = round(_confidence_from_word_count(word_count), 3)

    if confidence < confidence_floor:
        return None, confidence

    bf = BigFive(
        openness=round(_trait_score(tokens, "openness"), 3),
        conscientiousness=round(_trait_score(tokens, "conscientiousness"), 3),
        extraversion=round(_trait_score(tokens, "extraversion"), 3),
        agreeableness=round(_trait_score(tokens, "agreeableness"), 3),
        neuroticism=round(_trait_score(tokens, "neuroticism"), 3),
    )
    return bf, confidence
