from __future__ import annotations

import re


CRISIS_PATTERNS = [
    r"\bsuicid(?:e|al)\b",
    r"\bkill myself\b",
    r"\bend my life\b",
    r"\btake my life\b",
    r"\bwant to die\b",
    r"\bdon'?t want to live\b",
    r"\bself[-\s]?harm\b",
    r"\bhurt myself\b",
    r"\bno point (?:in )?(?:living|being alive)\b",
    r"\bdon'?t see the point\b",
]


def detect_crisis(message: str) -> bool:
    normalized = message.lower()
    return any(re.search(pattern, normalized) for pattern in CRISIS_PATTERNS)


def crisis_reply() -> str:
    return (
        "I'm really sorry you're feeling this much pain. Your safety matters right now: "
        "if you might hurt yourself or feel in immediate danger, please call or text 988 "
        "in the U.S. or contact local emergency services now. If you can, move near another "
        "person and tell them plainly that you need support."
    )

