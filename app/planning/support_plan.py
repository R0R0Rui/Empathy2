import re
from collections.abc import Mapping

from .schemas import SupportPlan


BASE_CONSTRAINTS = [
    "do not claim personal experience",
    "do not pretend to be human",
    "do not give generic reassurance only",
]

SELF_DISMISSAL_PATTERNS = [
    r"\bdramatic\b",
    r"\boverreact(?:ing|ed)?\b",
    r"\bi['’]?m fine\b",
    r"\bi am fine\b",
    r"\bwhatever\b",
    r"\bstupid\b",
    r"\bpathetic\b",
    r"\btoo sensitive\b",
    r"\bnot a big deal\b",
]


def _as_dict(understanding):
    if understanding is None:
        return {}
    if isinstance(understanding, Mapping):
        return dict(understanding)
    if hasattr(understanding, "model_dump"):
        return understanding.model_dump()
    if hasattr(understanding, "dict"):
        return understanding.dict()
    return {}


def _normalize(value):
    return str(value or "").lower()


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _append_unique(items, *values):
    for value in values:
        if value and value not in items:
            items.append(value)


def _contains_self_dismissal(user_message):
    text = _normalize(user_message)
    return any(re.search(pattern, text) for pattern in SELF_DISMISSAL_PATTERNS)


def _base_goal(data):
    support_need = _normalize(data.get("support_need"))
    primary_emotion = _normalize(data.get("primary_emotion")) or "the user's feeling"

    if support_need == "problem_solving":
        return "validate emotion, then support practical problem-solving"
    if support_need in {"safety_support", "resource_referral"}:
        return "support immediate safety and connection to appropriate help"
    if support_need == "grounding":
        return "help the user feel steadier and less alone"
    if support_need == "encouragement":
        return "recognize effort and offer grounded encouragement"
    if support_need == "normalization_without_dismissal":
        return "normalize the reaction without minimizing it"
    return f"validate and respond to {primary_emotion}"


def generate_support_plan(understanding: dict, user_message: str = "") -> SupportPlan:
    data = _as_dict(understanding)

    scenario_tier = _normalize(data.get("scenario_tier"))
    intent = _normalize(data.get("intent"))
    support_need = _normalize(data.get("support_need"))
    safety_flag = _normalize(data.get("safety_flag"))
    primary_emotion = _normalize(data.get("primary_emotion"))
    secondary_emotions = [_normalize(item) for item in _as_list(data.get("secondary_emotions"))]
    evidence = [str(item) for item in _as_list(data.get("evidence"))]
    crisis_signals = [str(item) for item in _as_list(data.get("crisis_signals"))]
    emotion_intensity = data.get("emotion_intensity", 0.0)
    is_implicit = bool(data.get("is_implicit", False))
    has_self_dismissal = _contains_self_dismissal(user_message)

    response_acts = []
    constraints = list(BASE_CONSTRAINTS)
    avoid = []
    safety_notes = []
    repair_priorities = []

    support_goal = _base_goal(data)

    _append_unique(
        response_acts,
        "name the likely emotion",
        "validate the user's reaction",
    )

    if primary_emotion in {"sadness", "grief", "loneliness"}:
        _append_unique(response_acts, "acknowledge loss or disconnection")
    if primary_emotion in {"anger", "annoyance"} or "anger" in secondary_emotions:
        _append_unique(response_acts, "validate anger without escalating blame")
    if primary_emotion in {"fear", "nervousness", "anxiety"} or "anxiety" in secondary_emotions:
        _append_unique(response_acts, "offer grounding before next steps")

    if scenario_tier == "subtle" or is_implicit:
        _append_unique(
            response_acts,
            "reflect implicit feeling",
            "validate without overstatement",
            "invite gentle exploration",
        )
        _append_unique(
            constraints,
            "do not overstate emotions the user did not name",
            "use tentative language for inferred feelings",
        )
        _append_unique(repair_priorities, "increase specificity to subtle cues")

    if has_self_dismissal:
        support_goal = "reduce self-dismissal and validate the underlying feeling"
        _append_unique(
            constraints,
            "do not agree with negative self-framing",
            "separate the user's feeling from self-judgment",
        )
        _append_unique(
            avoid,
            "agreeing that the user is dramatic or overreacting",
            "minimizing the concern because the user minimized it",
        )
        _append_unique(
            response_acts,
            "gently challenge self-dismissal",
            "validate the underlying hurt or stress",
        )
        _append_unique(repair_priorities, "address self-dismissal directly")

    if safety_flag in {"medium", "high"} or scenario_tier == "high_risk":
        support_goal = "support immediate safety and connection to appropriate help"
        _append_unique(
            response_acts,
            "provide safety-aware support",
            "encourage reaching out to trusted or emergency support",
            "reduce isolation",
        )
        _append_unique(
            constraints,
            "prioritize immediate safety over ordinary advice",
            "keep crisis language calm and direct",
        )
        _append_unique(
            safety_notes,
            "include crisis or emergency resources when risk may be immediate",
            "encourage the user to contact a trusted person or local emergency support",
        )
        for signal in crisis_signals:
            _append_unique(safety_notes, f"crisis signal: {signal}")
        _append_unique(
            avoid,
            "suggesting the user handle crisis feelings alone",
            "promising confidentiality in a safety crisis",
        )
        _append_unique(repair_priorities, "add concrete safety support")

    if intent == "advice_request":
        _append_unique(
            response_acts,
            "validate emotion before problem-solving",
            "offer one or two practical next steps",
        )
        _append_unique(
            constraints,
            "lead with emotional validation before advice",
            "keep advice collaborative rather than directive",
        )
        _append_unique(avoid, "jumping straight into instructions")

    if support_need == "clarifying_question":
        _append_unique(response_acts, "ask at most one clarifying question")
        _append_unique(constraints, "do not over-question")

    if evidence:
        _append_unique(
            repair_priorities,
            "ground response in evidence: " + "; ".join(evidence[:3]),
        )

    try:
        intensity = float(emotion_intensity)
    except (TypeError, ValueError):
        intensity = 0.0

    confidence = 0.55
    if data:
        confidence += 0.15
    if evidence:
        confidence += 0.1
    if scenario_tier:
        confidence += 0.05
    if intensity >= 0.7 or safety_flag in {"medium", "high"}:
        confidence += 0.05
    confidence = min(confidence, 0.95)

    return SupportPlan(
        support_goal=support_goal,
        response_acts=response_acts,
        constraints=constraints,
        avoid=avoid,
        safety_notes=safety_notes,
        repair_priorities=repair_priorities,
        confidence=round(confidence, 2),
    )
