"""
demo_understanding.py
=====================
Stand-alone demo of the LLM #1 Understanding layer. WITHOUT loading
any heavy ML models. Useful for:

  - Showing what the structured `understanding` output looks like
  - Reading examples in the deck/report
  - Sanity-checking changes to the analyzer

Run from repo root:
    python -m scripts.demo_understanding
"""

from __future__ import annotations
import json
from typing import Any

from app.understanding import UnderstandingAnalyzer


def _fake_emotions(label: str, conf: float) -> dict[str, Any]:
    """Synthesize a GoEmotions-style payload so we don't need the real model."""
    return {
        "primary":   {"label": label,    "confidence": conf},
        "secondary": {"label": "neutral","confidence": 0.10},
        "tertiary":  {"label": "neutral","confidence": 0.05},
        "top_emotions": [],
        "confidence_level": "high" if conf >= 0.5 else "low",
    }


# Each row: (user_message, simulated_GoEmotions_label, simulated_confidence)
SAMPLES = [
    ("I'm so happy today, I got the job!",                      "joy",         0.85),
    ("Yeah... I guess. Whatever.",                              "neutral",     0.60),
    ("I just stayed in bed today.",                             "neutral",     0.70),
    ("My boss said the project went well. I should be happy.",  "joy",         0.50),
    ("I'm so stressed. What should I do?",                      "nervousness", 0.75),
    ("Sometimes I think everyone would be better off without me.", "sadness",  0.65),
    ("I don't see the point. I want to die.",                   "sadness",     0.70),
    ("This homework is killing me, lol.",                       "annoyance",   0.50),
]


def main() -> None:
    analyzer = UnderstandingAnalyzer()
    print("=" * 72)
    print("Anchor LLM #1 — Understanding layer demo (mock GoEmotions input)")
    print("=" * 72)

    for i, (text, label, conf) in enumerate(SAMPLES, 1):
        emotions = _fake_emotions(label, conf)
        state = analyzer.analyze(user_message=text, history=[], emotions=emotions)
        print(f"\n[{i}] INPUT: {text!r}")
        print(f"    primary={state.primary_emotion!r}  intensity={state.emotion_intensity}")
        print(f"    valence={state.valence:+.2f}  arousal={state.arousal:.2f}")
        print(f"    intent={state.intent!r}  is_implicit={state.is_implicit}")
        print(f"    scenario_tier={state.scenario_tier!r}  safety_flag={state.safety_flag!r}")
        print(f"    support_need={state.support_need!r}")
        if state.evidence:
            print("    evidence:")
            for e in state.evidence:
                print(f"      - {e}")
        if state.crisis_signals:
            print(f"    crisis_signals={state.crisis_signals}")


if __name__ == "__main__":
    main()
