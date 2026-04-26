from __future__ import annotations

from collections.abc import Sequence

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


def build_user_content(
    emotion: str,
    message: str,
    history: Sequence[Turn] | None = None,
) -> str:
    history = history or []
    history_text = "\n".join(f"{turn['role']}: {turn['content']}" for turn in history)
    history_block = f"Conversation history:\n{history_text}\n\n" if history_text else ""
    return (
        f"Detected emotion profile: {emotion}\n\n"
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
