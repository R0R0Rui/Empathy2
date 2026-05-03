import json
import re


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


def _clean(value):
    if value is None:
        return ""
    return str(value).strip()


def _lower(value):
    return _clean(value).lower()


def _as_bool(value):
    if isinstance(value, bool):
        return value
    return _lower(value) in {"true", "1", "yes", "y"}


def _as_list(value):
    if value is None or value == "":
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


def _has_self_dismissal(user_message):
    text = _lower(user_message)
    return any(re.search(pattern, text) for pattern in SELF_DISMISSAL_PATTERNS)


def parse_understanding_json(raw_value):
    if raw_value is None or str(raw_value).strip() == "":
        return {}
    if isinstance(raw_value, dict):
        return raw_value
    try:
        parsed = json.loads(str(raw_value))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def build_support_plan(understanding, user_message=""):
    scenario_tier = _lower(understanding.get("scenario_tier"))
    support_need = _lower(understanding.get("support_need"))
    intent = _lower(understanding.get("intent"))
    safety_flag = _lower(understanding.get("safety_flag"))
    is_implicit = _as_bool(understanding.get("is_implicit"))
    crisis_signals = [str(item) for item in _as_list(understanding.get("crisis_signals"))]

    response_acts = ["validate the user's reaction"]
    constraints = [
        "do not claim personal experience",
        "do not pretend to be human",
        "do not give generic reassurance only",
    ]
    safety_notes = []
    repair_priorities = []

    support_goal = "provide grounded emotional support"

    if scenario_tier == "subtle" or is_implicit:
        support_goal = "reflect subtle emotional cues without overstatement"
        _append_unique(
            response_acts,
            "reflect implicit feeling",
            "validate without overstatement",
            "invite gentle exploration",
        )
        _append_unique(repair_priorities, "ground response in subtle cues")

    if _has_self_dismissal(user_message):
        support_goal = "reduce self-dismissal and validate the underlying feeling"
        _append_unique(
            constraints,
            "do not agree with negative self-framing",
            "separate the feeling from self-judgment",
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
            safety_notes,
            "include crisis or emergency resources if risk may be immediate",
            "encourage contact with a trusted person or local emergency support",
        )
        for signal in crisis_signals:
            _append_unique(safety_notes, f"crisis signal: {signal}")
        _append_unique(repair_priorities, "add concrete safety support")

    if intent == "advice_request" or support_need == "problem_solving":
        _append_unique(
            response_acts,
            "validate emotion before problem-solving",
            "offer one practical next step",
        )
        _append_unique(repair_priorities, "keep advice collaborative")

    confidence = 0.65
    if understanding:
        confidence += 0.1
    if scenario_tier:
        confidence += 0.05
    if safety_notes:
        confidence += 0.05

    return {
        "support_goal": support_goal,
        "response_acts": response_acts,
        "constraints": constraints,
        "safety_notes": safety_notes,
        "repair_priorities": repair_priorities,
        "confidence": round(min(confidence, 0.9), 2),
    }


def summarize_support_plan(plan):
    if not plan:
        return ""
    acts = ", ".join(plan.get("response_acts", [])[:3])
    return f"{plan.get('support_goal', '')}: {acts}".strip(": ")
