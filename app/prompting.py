from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from app.session import Turn


SYSTEM_PROMPT = (
    "You are a warm and empathetic AI assistant. "
    "Always respond with emotional support first. "
    "Acknowledge and validate the user's feelings before asking questions. "
    "Keep responses 2-4 sentences."
)


def format_history(history: Sequence[Turn]) -> str:
    return "\n".join(f"{turn['role']}: {turn['content']}" for turn in history)


def build_classifier_input(message: str, history: Sequence[Turn]) -> str:
    history_text = format_history(history)
    parts: list[str] = []
    if history_text:
        parts.extend(["[HISTORY]", history_text])
    parts.extend(["[CURRENT]", f"user: {message}"])
    return "\n".join(parts)


def format_emotion_profile(emotions: dict[str, dict[str, float | str]]) -> str:
    primary = emotions["primary"]
    secondary = emotions["secondary"]
    tertiary = emotions["tertiary"]
    return (
        f"{primary['label']} {float(primary['confidence']):.2f}, "
        f"{secondary['label']} {float(secondary['confidence']):.2f}, "
        f"{tertiary['label']} {float(tertiary['confidence']):.2f}"
    )


def format_understanding_profile(understanding: dict[str, Any] | None) -> str:
    """
    Render the LLM #1 Understanding payload as a compact prompt fragment for
    the response generator (LLM #2).

    This is what makes the structured `understanding` field actually
    INFLUENCE the reply — not just be metadata. We expose only the
    fields LLM #2 should respect (intent, scenario tier, support need,
    safety level) plus the emotion summary. We do NOT leak
    `pii_redacted_text` or `layer_outputs` into the prompt.
    """
    if not understanding:
        return ""

    secondary = ", ".join(understanding.get("secondary_emotions") or []) or "none"
    guidance = "; ".join(understanding.get("evidence") or []) or "none"
    crisis = ", ".join(understanding.get("crisis_signals") or []) or "none"

    return (
        "Structured understanding from LLM #1:\n"
        f"- primary emotion: {understanding.get('primary_emotion', 'unknown')}\n"
        f"- secondary emotions: {secondary}\n"
        f"- emotion intensity: {float(understanding.get('emotion_intensity', 0.0)):.2f}\n"
        f"- valence: {float(understanding.get('valence', 0.0)):+.2f}  "
        f"arousal: {float(understanding.get('arousal', 0.5)):.2f}\n"
        f"- intent: {understanding.get('intent', 'unknown')}\n"
        f"- scenario tier: {understanding.get('scenario_tier', 'common')}  "
        f"(implicit={bool(understanding.get('is_implicit', False))})\n"
        f"- safety flag: {understanding.get('safety_flag', 'none')}\n"
        f"- crisis signals: {crisis}\n"
        f"- support need: {understanding.get('support_need', 'validation')}\n"
        f"- evidence: {guidance}\n"
    )


def build_user_content(
    emotion: str,
    message: str,
    history: Sequence[Turn] | None = None,
    understanding: dict[str, Any] | None = None,
) -> str:
    history = history or []
    history_text = "\n".join(f"{turn['role']}: {turn['content']}" for turn in history)
    history_block = f"Conversation history:\n{history_text}\n\n" if history_text else ""
    understanding_block = format_understanding_profile(understanding)
    if understanding_block:
        understanding_block += "\n"
    return (
        f"Detected emotion profile: {emotion}\n\n"
        + understanding_block
        + history_block
        + f"User message:\n<user>{message}</user>\n\n"
        + "Respond with emotional validation first, then gentle support. "
        "If multiple emotions are present, acknowledge the main emotion "
        "and reflect the others naturally."
    )


def build_generation_messages(
    message: str,
    history: Sequence[Turn],
    emotions: dict[str, dict[str, float | str]],
) -> list[dict[str, str]]:
    emotion_profile = format_emotion_profile(emotions)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_content(emotion_profile, message, history)},
    ]


# --- LLM #1 Understanding additions ---------------------------------
def build_generation_messages_with_understanding(
    message: str,
    history: Sequence[Turn],
    emotions: dict[str, dict[str, float | str]],
    understanding: dict[str, Any] | None,
) -> list[dict[str, str]]:
    """
    Same as `build_generation_messages` but also injects the Understanding
    payload into the user-content prompt so LLM #2 can read it.

    Falls back to the original behavior if `understanding` is None.
    """
    if understanding is None:
        return build_generation_messages(message, history, emotions)

    emotion_profile = format_emotion_profile(emotions)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_content(emotion_profile, message, history, understanding),
        },
    ]
