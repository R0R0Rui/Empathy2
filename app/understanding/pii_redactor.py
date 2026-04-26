"""
pii_redactor.py
===============
Redacts personally-identifiable information from user text BEFORE it gets
written to logs, screenshots, or any persistent artifact.

WHY THIS LIVES IN INPUT PROCESSING:
  Users often paste phone numbers, emails, or addresses into emotional
  messages ("my number is 555-1234, please tell my mom..."). Logging that
  raw text creates privacy risk. We replace it with placeholders for
  logging, but the original text is still passed to the classifier and
  generator (so the model can read it and respond appropriately).

WHAT WE REDACT:
  - Email addresses    -> [EMAIL]
  - Phone numbers      -> [PHONE]   (US-style + international fragments)
  - URLs               -> [URL]

WHAT WE DELIBERATELY DO NOT REDACT (yet):
  - First names — not safe to detect with regex; needs an NER model
  - Addresses — too varied
  These can be added later if the team adopts spaCy NER or similar.
"""

from __future__ import annotations
import re

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
# Phone: 7+ digits possibly with separators, optionally with country code.
_PHONE_RE = re.compile(r"(?:\+?\d[\d\-\s().]{7,}\d)")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")


def redact_pii(text: str) -> str:
    """Return a logged-safe version of `text` with PII replaced."""
    if not text:
        return ""
    out = _EMAIL_RE.sub("[EMAIL]", text)
    out = _URL_RE.sub("[URL]", out)
    out = _PHONE_RE.sub("[PHONE]", out)
    return out
