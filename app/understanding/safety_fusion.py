"""
safety_fusion.py
================
Combines the team's existing app/safety.py crisis detector with our own
extended crisis lexicon, then escalates the safety_flag with one-way rule:
ANY layer can ESCALATE; no layer can DOWNGRADE another's escalation.

WHY WE WRAP rather than REPLACE app/safety.py:
  - The team's `detect_crisis()` is already used by /chat to short-circuit
    to crisis_reply(). We must keep its behaviour intact so existing
    integration tests still pass.
  - But the team's list (10 patterns) misses well-documented passive
    suicidal ideation phrases ("everyone would be better off without me",
    "no way out", "want it to stop"). Those are clinically important.
  - So we ADD an extended list and combine. The team's list keeps full
    authority over `crisis_detected: bool` in the API. Our additions feed
    into `safety_flag` (none/low/medium/high), which is a richer signal.

OUTPUT FROM THIS MODULE:
    SafetyVerdict {
        crisis_detected_team: bool,    # what app/safety.py says (unchanged)
        crisis_detected_extended: bool # our extended check
        safety_flag: str               # none/low/medium/high
        crisis_signals: list[str]      # human-readable reason strings
    }
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import List

from app.safety import detect_crisis as team_detect_crisis


_REPO_ROOT = Path(__file__).resolve().parents[2]
LEXICON_PATH = _REPO_ROOT / "data" / "lexicons" / "crisis_keywords.txt"


@dataclass
class SafetyVerdict:
    crisis_detected_team: bool = False
    crisis_detected_extended: bool = False
    safety_flag: str = "none"           # none | low | medium | high
    crisis_signals: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Extended lexicon loader
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_extended_lexicon() -> dict[str, List[str]]:
    """Parse data/lexicons/crisis_keywords.txt into {category: [phrases]}."""
    if not LEXICON_PATH.exists():
        return {}
    by_cat: dict[str, List[str]] = {}
    current = "UNCATEGORIZED"
    with open(LEXICON_PATH, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("# CATEGORY:"):
                current = line.split(":", 1)[1].strip()
                by_cat.setdefault(current, [])
                continue
            if line.startswith("#"):
                continue
            by_cat.setdefault(current, []).append(line.lower())
    return by_cat


def _scan_extended(text: str) -> dict[str, List[str]]:
    """Return {category: [matched phrases]} for the extended lexicon."""
    text_lc = text.lower()
    matches: dict[str, List[str]] = {}
    for category, phrases in _load_extended_lexicon().items():
        for phrase in phrases:
            if " " in phrase:
                if phrase in text_lc:
                    matches.setdefault(category, []).append(phrase)
            else:
                if re.search(r"\b" + re.escape(phrase) + r"\b", text_lc):
                    matches.setdefault(category, []).append(phrase)
    return matches


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def fuse_safety(text: str) -> SafetyVerdict:
    """Return a fused safety verdict (team's check + extended)."""
    team_flag = team_detect_crisis(text)
    extended_matches = _scan_extended(text)

    # Decide safety_flag
    flag = "none"
    signals: List[str] = []

    if team_flag:
        flag = "high"
        signals.append("team_safety_crisis_pattern_match")

    if extended_matches:
        cats = set(extended_matches.keys())
        # Suicidal-ideation matches always escalate to high.
        if "SUICIDAL_IDEATION" in cats or "SELF_HARM" in cats:
            flag = "high"
        elif flag == "none":
            flag = "medium"
        for cat, terms in extended_matches.items():
            signals.append(f"{cat}: {', '.join(sorted(set(terms)))}")

    return SafetyVerdict(
        crisis_detected_team=team_flag,
        crisis_detected_extended=bool(extended_matches),
        safety_flag=flag,
        crisis_signals=signals,
    )
