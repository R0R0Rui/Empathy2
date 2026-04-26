"""
intensity.py
============
Compute emotion_intensity in [0, 1].

WHY THIS IS DIFFERENT FROM CLASSIFIER CONFIDENCE:
  GoEmotions returns the model's *confidence* that a label is correct.
  But "I'm a little tired" and "I am UTTERLY DESTROYED right now" can both
  return high confidence in 'sadness' — the classifier is sure they're
  sadness, but they're very different intensities.

  We blend two signals to estimate the *intensity*:
    - The top emotion's confidence (weight 0.5)
    - The density of emotion words (Layer A) per total words (weight 0.3)
    - Punctuation cues — exclamation marks, ALL CAPS, repetition (weight 0.2)
"""

from __future__ import annotations
import re

from .lexical_features import LexicalFeatures


def _punctuation_intensity(text: str) -> float:
    """Heuristic: how 'loud' is the text? Returns [0, 1]."""
    if not text:
        return 0.0
    score = 0.0

    # Multiple exclamations
    excl = text.count("!")
    score += min(0.5, excl * 0.15)

    # ALL CAPS words (3+ chars)
    caps_words = re.findall(r"\b[A-Z]{3,}\b", text)
    if caps_words:
        score += min(0.5, len(caps_words) * 0.2)

    # Character repetition (e.g., "soooo tired")
    if re.search(r"([a-zA-Z])\1{2,}", text):
        score += 0.2

    return min(1.0, score)


def compute_emotion_intensity(
    classifier_top_confidence: float,
    lex: LexicalFeatures,
    raw_text: str,
) -> float:
    """
    Blend three signals into a single intensity score.

    All inputs are in [0, 1]. Output is in [0, 1] rounded to 3 decimals.
    """
    # Density of emotion words, capped
    if lex.total_word_count == 0:
        density = 0.0
    else:
        density = min(
            1.0,
            sum(lex.emotion_word_counts.values()) / max(lex.total_word_count, 1) * 4.0,
        )

    punct = _punctuation_intensity(raw_text)

    blended = 0.5 * classifier_top_confidence + 0.3 * density + 0.2 * punct
    return round(min(1.0, max(0.0, blended)), 3)
