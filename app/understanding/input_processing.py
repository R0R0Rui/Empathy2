"""
input_processing.py
===================
Input Processing layer of the Anchor LLM #1 pipeline.

The proposal architecture box:

    User Input → INPUT PROCESSING → Emotion + Intent Understanding (LLM #1)

is implemented across these helpers — this file is a thin FACADE that ties
them together so callers (e.g. `analyzer.py`) and readers (e.g. teammates,
the professor) have a single named entry point for "Input Processing".

What "Input Processing" does in our system:
  1. Text normalization
       - Trim whitespace, normalize Unicode (NFKC), keep emotional
         punctuation and emoji untouched (they carry signal).
  2. PII redaction (for logs / screenshots only — the model still sees raw)
       - Email, phone, URL → [EMAIL] / [PHONE] / [URL]
       (`pii_redactor.py`)
  3. Lexical feature extraction (LIWC-style word counting)
       - Counts in psychologically meaningful categories: sadness, anxiety,
         anger, joy, etc., plus continuous valence/arousal scores.
       (`lexical_features.py`)
  4. Safety prefilter
       - Combines team's app/safety.py crisis detector with our extended
         passive-ideation lexicon. One-way escalation rule.
       (`safety_fusion.py`)

Public function:
    process_input(user_message) -> ProcessedInput
"""

from __future__ import annotations
import re
import unicodedata
from dataclasses import dataclass

from .lexical_features import LexicalFeatures, extract_lexical_features
from .pii_redactor import redact_pii
from .safety_fusion import SafetyVerdict, fuse_safety


@dataclass(frozen=True)
class ProcessedInput:
    """All the input-processing outputs bundled together."""
    raw_text: str               # exactly what the user typed
    normalized_text: str        # whitespace + Unicode normalized
    pii_redacted_text: str      # safe to log
    lexical: LexicalFeatures    # LIWC-style counts + valence/arousal
    safety: SafetyVerdict       # safety prefilter result


_WHITESPACE_RE = re.compile(r"[ \t]+")


def normalize_text(text: str) -> str:
    """Light normalization that preserves emotional cues (caps, emoji, punctuation)."""
    if not text:
        return ""
    out = unicodedata.normalize("NFKC", text)
    # Strip and collapse internal runs of spaces/tabs, but keep newlines.
    paragraphs = [_WHITESPACE_RE.sub(" ", p).strip() for p in out.split("\n")]
    return "\n".join(p for p in paragraphs if p)


def process_input(user_message: str) -> ProcessedInput:
    """The single Input Processing entry point used by analyzer.py."""
    normalized = normalize_text(user_message)
    return ProcessedInput(
        raw_text=user_message,
        normalized_text=normalized,
        pii_redacted_text=redact_pii(normalized),
        lexical=extract_lexical_features(normalized),
        safety=fuse_safety(normalized),
    )
