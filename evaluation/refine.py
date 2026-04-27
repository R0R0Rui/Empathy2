
FAILURE_REWRITE_RULES = {
    "self_experience_claim": (
        "Remove any claim that the AI has personal experience. "
        "Validate the user's feeling without pretending to share it."
    ),
    "too_short": (
        "Expand slightly with specific emotional validation and one gentle next step."
    ),
    "agrees_with_negative_framing": (
        "Do not agree with the user's negative framing. Gently reframe it."
    ),
    "misses_self_dismissal": (
        "Address the user's self-dismissal directly and gently. Normalize the feeling."
    ),
    "lacks_safety_support": (
        "Add gentle safety-aware support and encourage reaching out to someone trusted."
    ),
}


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
