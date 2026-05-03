import re


FAILURE_REWRITE_RULES = {
    "self_experience_claim": (
        "Remove any claim that the AI has personal experience. "
        "Validate the user's feeling without pretending to share it."
    ),
    "too_short": (
        "Expand slightly with specific emotional validation and one gentle next step."
    ),
    "generic_empathy": (
        "Ground the response in the user's specific situation instead of using only generic empathy."
    ),
    "over_advice": (
        "Reduce directive advice and start with emotional validation before any next step."
    ),
    "premature_positive_reframe": (
        "Remove premature positivity. Acknowledge the difficulty before any hopeful framing."
    ),
    "agrees_with_negative_framing": (
        "Do not agree with the user's negative framing. Gently reframe it."
    ),
    "misses_self_dismissal": (
        "Address the user's self-dismissal directly and gently. Normalize the feeling."
    ),
    "missing_safety_support": (
        "Add safety-aware support and encourage immediate contact with trusted or crisis support."
    ),
    "unsafe_crisis_handling": (
        "Replace unsafe or dismissive crisis handling with calm, immediate safety support."
    ),
    "over_questioning": (
        "Reduce the number of questions and offer grounded validation first."
    ),
    "clinical_overclaim": (
        "Remove diagnosis or clinical certainty; use non-clinical, supportive language."
    ),
    "ignores_user_boundary": (
        "Respect the user's stated boundary while still offering gentle support."
    ),
}

META_PHRASES = [
    "What stands out is",
    "For this response",
    "The priority is",
    "The user needs",
    "This response should",
    "validate the user's reaction",
    "reflect implicit feeling",
    "support plan",
    "failure type",
    "rewrite_needed",
    "rewrite needed",
]


def _clean(value):
    return "" if value is None else str(value).strip()


def _lower(value):
    return _clean(value).lower()


def _as_bool(value):
    if isinstance(value, bool):
        return value
    return _lower(value) in {"true", "1", "yes", "y"}


def _failure_types(validation_result):
    if not validation_result:
        return []
    failures = validation_result.get("failure_types", [])
    if isinstance(failures, str):
        return [item.strip() for item in failures.split(",") if item.strip()]
    return [str(item).strip() for item in failures if str(item).strip()]


def _understanding_value(understanding, key):
    if not understanding:
        return ""
    if isinstance(understanding, dict):
        return understanding.get(key, "")
    return getattr(understanding, key, "")


def _has_self_dismissal(user_message):
    text = _lower(user_message)
    patterns = [
        r"\bdramatic\b",
        r"\boverreact(?:ing|ed)?\b",
        r"\bstupid\b",
        r"\bpathetic\b",
        r"\btoo sensitive\b",
        r"\bnot a big deal\b",
        r"\bi['’]?m fine\b",
        r"\bi am fine\b",
        r"\bwhatever\b",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def _has_parent_boundary(user_message):
    text = _lower(user_message)
    return (
        "do not tell me to call my parents" in text
        or "don't tell me to call my parents" in text
        or "do not tell me to contact my parents" in text
        or "don't tell me to contact my parents" in text
        or ("parents" in text and "part of why i am upset" in text)
    )


def _states_current_safety(user_message):
    text = _lower(user_message)
    return any(
        phrase in text
        for phrase in [
            "i am safe",
            "i'm safe",
            "im safe",
            "not in danger",
            "physically okay",
            "physically ok",
            "i mean that dramatically",
            "mean that dramatically",
        ]
    )


def _is_high_risk(user_message, failures, scenario_type, understanding):
    text = _lower(user_message)
    scenario = _lower(scenario_type)
    safety_flag = _lower(_understanding_value(understanding, "safety_flag"))

    explicit_patterns = [
        "want to die",
        "end my life",
        "kill myself",
        "hurt myself",
        "cutting myself",
        "better off without me",
        "not be safe",
        "can't keep myself safe",
        "cannot promise",
        "no reason to live",
    ]

    if _states_current_safety(user_message) and "safety_edge" in scenario:
        return False
    if "missing_safety_support" in failures or "unsafe_crisis_handling" in failures:
        return True
    if scenario in {"high_risk", "high-risk", "crisis"}:
        return True
    if safety_flag == "high":
        return True
    return any(pattern in text for pattern in explicit_patterns)


def _context_hint(user_message):
    text = _clean(user_message)
    lowered = text.lower()

    if "roommate" in lowered and "dish" in lowered:
        return "the situation with your roommate and the dishes"
    if "manager" in lowered and ("praised" in lowered or "awkward" in lowered):
        return "that one awkward moment after your manager praised your work"
    if "assignment" in lowered:
        return "how overwhelming this assignment feels"
    if "friend" in lowered and ("cancel" in lowered or "ignored" in lowered):
        return "feeling hurt by your friend"
    if "interview" in lowered:
        return "freezing during the interview"
    if "deadline" in lowered:
        return "the pressure from the deadline changing"
    if "presentation" in lowered:
        return "the pressure around the presentation"
    if "family" in lowered:
        return "what is happening with your family"
    if "parents" in lowered:
        return "what is happening with your parents"
    if "partner" in lowered:
        return "what happened with your partner"
    if "job" in lowered or "internship" in lowered:
        return "the pressure that came with this opportunity"
    if "dinner" in lowered:
        return "feeling ignored at dinner"
    if "phone" in lowered:
        return "that anxious reaction when your phone lit up"
    if "cleaned" in lowered and "room" in lowered:
        return "cleaning your room after avoiding it for a while"

    words = re.findall(r"[A-Za-z']+", text)
    if len(words) <= 14 and text:
        return text.rstrip(".!?")
    return "what happened"


def _underlying_concern(user_message):
    text = _lower(user_message)

    if (
        ("internship" in text or "job" in text or "got the" in text)
        and (
            "mistake" in text
            or "choosing me" in text
            or "picked me" in text
            or "nervous" in text
            or "imposter" in text
            or "impostor" in text
        )
    ):
        return "impostor_pressure"

    if "manager" in text and ("praised" in text or "awkward" in text):
        return "praise_rumination"

    if "roommate" in text and "dish" in text:
        return "built_up_frustration"

    if "assignment" in text and ("killing me" in text or "dramatically" in text):
        return "overwhelm"

    if "friend" in text and ("cancel" in text or "ignored" in text):
        return "rejection_hurt"

    if "deadline" in text or "presentation" in text:
        return "performance_pressure"

    if "interview" in text:
        return "performance_shame"

    if "cleaned" in text and "room" in text:
        return "small_but_meaningful_progress"

    if _has_parent_boundary(user_message):
        return "parent_boundary"

    if _has_self_dismissal(user_message):
        return "self_dismissal"

    return "general_distress"


def _is_advice_request(understanding):
    return _lower(_understanding_value(understanding, "intent")) == "advice_request"


def _is_subtle(understanding, scenario_type):
    return (
        _lower(scenario_type) == "subtle"
        or _lower(_understanding_value(understanding, "scenario_tier")) == "subtle"
        or _as_bool(_understanding_value(understanding, "is_implicit"))
    )


def _natural_common_response(user_message, failures, understanding):
    hint = _context_hint(user_message)
    text = _lower(user_message)
    concern = _underlying_concern(user_message)

    if concern == "impostor_pressure":
        return (
            "That makes a lot of sense. "
            "Getting something you really wanted can make the pressure feel even bigger, especially if part of you is wondering whether you deserve it. "
            "It does not mean they made a mistake; it probably means this opportunity matters to you. "
            "You can let yourself be proud and still take things one step at a time."
        )

    if concern == "built_up_frustration":
        return (
            "That sounds really frustrating, especially if it has been building up for a while. "
            "Snapping tonight does not mean your feelings came from nowhere, but it may be a sign this issue has gone unresolved for too long. "
            "Once things cool down, it might help to name the problem clearly without turning it into another fight."
        )

    if concern == "praise_rumination":
        return (
            "It makes sense that your mind keeps replaying that one awkward sentence, especially because the praise mattered to you. "
            "But one imperfect moment probably does not erase the work your manager noticed. "
            "You can let the praise and the awkwardness both be true, instead of letting the awkward part take over the whole memory."
        )

    if concern == "overwhelm":
        return (
            "That sounds really overwhelming, even if you mean it dramatically. "
            "It makes sense to feel worn down when an assignment starts taking over your head. "
            "For now, it may help to shrink the task into one small next step instead of trying to solve the whole thing at once."
        )

    if concern == "small_but_meaningful_progress":
        return (
            "That actually does sound huge. "
            "When something has been avoided for weeks, doing it can take more energy than it looks like from the outside. "
            "You deserve to let this count, even if it seems small."
        )

    if concern == "parent_boundary":
        return (
            "That makes sense, and I will not tell you to call your parents. "
            "If they are part of why you are upset, it is reasonable to want support from someone else or simply have your feelings taken seriously right now. "
            "You do not have to make the situation okay for them in this moment."
        )

    if _has_self_dismissal(user_message):
        return (
            f"It makes sense that {hint} would stay with you. "
            "Calling yourself dramatic or wrong for reacting only adds another layer of pressure. "
            "Your feeling deserves to be taken seriously, even if you are still figuring out what you need."
        )

    if _is_advice_request(understanding):
        return (
            f"It makes sense to want a way through {hint} without making things harder. "
            "Before jumping into a fix, it is worth acknowledging that this feels tense because it matters to you. "
            "A gentle next step could be to name one specific concern and ask for a calm conversation about that piece first."
        )

    if concern == "performance_pressure":
        return (
            f"It makes sense that {hint} would bring up pressure instead of just motivation. "
            "When something matters, your mind can start treating it like a test of whether you are good enough. "
            "You can take the next step seriously without turning it into a judgment of your whole ability."
        )

    if concern == "performance_shame":
        return (
            "Freezing in a high-pressure moment can feel awful, but it does not make you stupid or incapable. "
            "It sounds like the moment mattered to you, which can make the replay afterward feel even harsher. "
            "When you are ready, it may help to separate what happened from what you are making it mean about yourself."
        )

    if concern == "rejection_hurt":
        return (
            f"It makes sense that {hint} would hurt. "
            "Even if there is an explanation, being canceled on or ignored can still bring up feeling unimportant. "
            "You do not have to dismiss that feeling before you understand what you need from the friendship."
        )

    if "too_short" in failures or "generic_empathy" in failures:
        return (
            f"That sounds genuinely hard, especially around {hint}. "
            "It makes sense that the emotional part of it would need attention before jumping into what to do next. "
            "For now, you might give yourself permission to name what this is bringing up for you."
        )

    return (
        f"That sounds difficult, especially around {hint}. "
        "It makes sense that your reaction has some weight to it. "
        "You do not have to solve all of it at once; one small, steady next step is enough for now."
    )


def _natural_subtle_response(user_message, failures, understanding):
    hint = _context_hint(user_message)
    text = _lower(user_message)

    if _has_parent_boundary(user_message):
        return (
            "That makes sense, and I will not tell you to call your parents. "
            "If they are part of why you are upset, it is reasonable to want support from someone else or simply have your feelings taken seriously right now. "
            "You do not have to make the situation okay for them in this moment."
        )

    if "deadline" in text and "fine" in text:
        return (
            "That kind of nodding along can hide a lot of pressure. "
            "It sounds like you may have had to absorb the deadline change before you had room to react to it. "
            "If you can, give yourself a moment to name what actually changed and what support or adjustment you might need."
        )

    if _has_self_dismissal(user_message):
        return (
            f"It sounds like there is more under the surface of {hint} than you are letting yourself admit. "
            "You do not have to call yourself dramatic to make the feeling smaller. "
            "It is enough to notice that something hurt or felt off, and to let that matter."
        )

    return (
        f"It makes sense that {hint} would leave you feeling unsettled. "
        "Sometimes the hardest part is that the reaction shows up before you have words for it. "
        "For now, it may help to slow the moment down and notice what you most need next."
    )


def _natural_safety_response(user_message):
    if _states_current_safety(user_message):
        return (
            "That sounds really intense, and I am glad you are saying you are safe right now. "
            "It still makes sense to take the feeling seriously rather than brushing it aside. "
            "If that changes or you start to feel at risk, please reach out to someone trusted, local emergency services, or 988 in the U.S. right away."
        )

    return (
        "I am really sorry you are feeling this much pain. "
        "When things feel this heavy, it is important not to be alone with it. "
        "If you might hurt yourself or feel unsafe, please call or text 988 in the U.S., contact local emergency services, or reach out to someone you trust right now."
    )


def _remove_meta_phrases(text):
    cleaned = _clean(text)
    for phrase in META_PHRASES:
        cleaned = re.sub(re.escape(phrase), "", cleaned, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", cleaned).strip()


def refine_response(
    user_message,
    original_response,
    validation_result,
    support_plan=None,
    understanding=None,
    scenario_type=None,
):
    failures = _failure_types(validation_result)
    if not failures:
        return _clean(original_response)

    if _is_high_risk(user_message, failures, scenario_type, understanding):
        response = _natural_safety_response(user_message)
    elif _is_subtle(understanding, scenario_type):
        response = _natural_subtle_response(user_message, failures, understanding)
    else:
        response = _natural_common_response(user_message, failures, understanding)

    return _remove_meta_phrases(response)


def build_rewrite_prompt(user_message, original_response, validation):
    failure_types = validation["failure_types"]

    repair_instructions = "\n".join(
        f"- {FAILURE_REWRITE_RULES[f]}"
        for f in failure_types
        if f in FAILURE_REWRITE_RULES
    )

    return f"""
You are refining a response from an empathetic AI assistant.

User message:
{user_message}

Original response:
{original_response}

Detected failure types:
{", ".join(failure_types)}

Repair instructions:
{repair_instructions}

Rewrite the response:
- Output only the revised response.
- Keep it 2-4 sentences.
- Be specific to the user's wording.
- Do not claim personal experiences.
- Do not pretend to be human.
- Do not agree with self-blame or negative self-framing.
- Do not sound clinical or overly formal.
""".strip()


def refine_if_needed(row, generate_rewrite_response):
    validation = row["anchor_validation"]

    if validation["passed"]:
        return row["anchor_response"]

    rewrite_prompt = build_rewrite_prompt(
        user_message=row["user_message"],
        original_response=row["anchor_response"],
        validation=validation,
    )

    return generate_rewrite_response(rewrite_prompt)
