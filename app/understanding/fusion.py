"""
fusion.py
=========
Combine all the upstream signals into a single UnderstandingState.

WHY A SEPARATE FUSION STEP:
  Each signal source is good at different things and bad at others:
    - GoEmotions (Layer B): great categorical labels, weak intensity, weak
      on implicit messages.
    - Lexical features (Layer A): interpretable, hard to fool on subtle
      cases, but coarse.
    - Safety: must always have veto power on tier and intent.
    - Intent classifier: rule-based, good for surface cues; sometimes wrong
      on ambiguous text.
  Fusion explicitly composes these into the final EmotionState. Putting
  it in one file makes the decision logic auditable.

KEY RULE:
  Safety is one-way — any layer can ESCALATE the safety_flag, but no layer
  can DOWNGRADE another's escalation. This is what protects us from a
  classifier that confidently mis-labels a crisis message as 'neutral'.
"""

from __future__ import annotations
from typing import List

from .schemas import UnderstandingState, BigFive
from .lexical_features import LexicalFeatures
from .safety_fusion import SafetyVerdict
from .intensity import compute_emotion_intensity
from .implicit_detector import is_implicit_emotion
from .scenario_tier import assign_scenario_tier
from .intent_classifier import classify_intent, normalize_intent
from .support_need import pick_support_need, normalize_support_need


def _build_evidence(
    lex: LexicalFeatures,
    classifier_top_label: str,
    classifier_top_confidence: float,
    is_implicit: bool,
    safety: SafetyVerdict,
) -> List[str]:
    """Short, audit-friendly reasoning trail."""
    evidence: List[str] = []
    if safety.safety_flag != "none":
        evidence.append(
            f"Safety pre-screen flagged: {safety.safety_flag}"
            + (f" — {safety.crisis_signals[0]}" if safety.crisis_signals else "")
        )
    if is_implicit:
        evidence.append(
            "User implies an emotion through context or behavior rather than naming it."
        )
    else:
        if lex.explicit_emotion_words:
            evidence.append(
                "User explicitly named feeling with words: "
                + ", ".join(sorted(set(lex.explicit_emotion_words))[:3])
            )
        elif classifier_top_label != "neutral":
            evidence.append(
                f"GoEmotions classifier confident in '{classifier_top_label}' "
                f"({classifier_top_confidence:.2f})."
            )
    if lex.emotion_word_counts:
        top_lex = sorted(lex.emotion_word_counts.items(), key=lambda kv: -kv[1])[:2]
        evidence.append("Lexical signals: " + ", ".join(f"{k}={v}" for k, v in top_lex))
    return evidence[:3]


def fuse_into_understanding_state(
    user_text: str,
    pii_redacted_text: str,
    classifier_emotions: dict,
    lex: LexicalFeatures,
    safety: SafetyVerdict,
    personality: BigFive | None,
    personality_confidence: float,
) -> UnderstandingState:
    """
    Single fusion entry. This function is the only place where 'business
    rules' for combining layer outputs live. Keep it boring and readable.
    """
    # Pull out the team's ranked emotions (primary/secondary/tertiary)
    primary = classifier_emotions["primary"]
    secondary = classifier_emotions["secondary"]
    tertiary = classifier_emotions["tertiary"]

    primary_label = str(primary["label"])
    primary_conf = float(primary["confidence"])

    secondary_emotions: List[str] = [
        str(secondary["label"]),
        str(tertiary["label"]),
    ]
    # De-dup, drop neutrals, drop the primary label
    secondary_emotions = [
        e for e in secondary_emotions
        if e and e != primary_label and e != "neutral"
    ]

    # Implicitness drives subtle tier
    is_implicit = is_implicit_emotion(
        lex=lex,
        classifier_top_label=primary_label,
        classifier_top_confidence=primary_conf,
        raw_text=user_text,
    )

    # Tier respects safety first
    tier = assign_scenario_tier(safety.safety_flag, is_implicit)

    # Intent — uses the team's safety detector for the crisis short-circuit
    intent_raw = classify_intent(
        user_text=user_text,
        primary_emotion=primary_label,
        crisis_detected=safety.crisis_detected_team or safety.safety_flag in {"high", "medium"},
    )
    intent = normalize_intent(intent_raw)

    # Intensity = blend of confidence + lex density + punctuation cues
    intensity = compute_emotion_intensity(
        classifier_top_confidence=primary_conf,
        lex=lex,
        raw_text=user_text,
    )

    # Valence / arousal — from VAD lexicon
    valence = lex.valence_score
    arousal = lex.arousal_score

    # Support strategy for LLM #2
    support = pick_support_need(
        primary_emotion=primary_label,
        intent=intent,
        safety_flag=safety.safety_flag,
        is_implicit=is_implicit,
    )
    support = normalize_support_need(support)

    # Audit trail
    evidence = _build_evidence(
        lex=lex,
        classifier_top_label=primary_label,
        classifier_top_confidence=primary_conf,
        is_implicit=is_implicit,
        safety=safety,
    )

    return UnderstandingState(
        primary_emotion=primary_label,
        secondary_emotions=secondary_emotions,
        emotion_intensity=intensity,
        valence=valence,
        arousal=arousal,
        intent=intent,
        is_implicit=is_implicit,
        scenario_tier=tier,
        safety_flag=safety.safety_flag,
        crisis_signals=safety.crisis_signals,
        support_need=support,
        evidence=evidence,
        personality=personality,
        personality_confidence=personality_confidence,
        pii_redacted_text=pii_redacted_text,
        layer_outputs={
            "lexical": {
                "emotion_word_counts": lex.emotion_word_counts,
                "valence": lex.valence_score,
                "arousal": lex.arousal_score,
                "explicit_words": lex.explicit_emotion_words,
                "total_words": lex.total_word_count,
            },
            "classifier": {
                "primary": dict(primary),
                "secondary": dict(secondary),
                "tertiary": dict(tertiary),
            },
            "safety": {
                "team_detected": safety.crisis_detected_team,
                "extended_detected": safety.crisis_detected_extended,
                "signals": safety.crisis_signals,
            },
        },
    )
