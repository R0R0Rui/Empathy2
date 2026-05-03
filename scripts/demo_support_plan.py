import json

from app.planning import generate_support_plan


EXAMPLES = [
    {
        "name": "subtle self-dismissal",
        "user_message": "I am probably being dramatic, but I felt ignored at dinner.",
        "understanding": {
            "primary_emotion": "sadness",
            "secondary_emotions": ["loneliness"],
            "emotion_intensity": 0.62,
            "intent": "venting",
            "is_implicit": True,
            "scenario_tier": "subtle",
            "safety_flag": "none",
            "support_need": "validation_and_gentle_exploration",
            "evidence": ["being dramatic", "felt ignored"],
            "crisis_signals": [],
        },
    },
    {
        "name": "high-risk safety",
        "user_message": "I cannot promise I will be safe if I stay alone tonight.",
        "understanding": {
            "primary_emotion": "fear",
            "secondary_emotions": ["sadness"],
            "emotion_intensity": 0.9,
            "intent": "crisis_or_safety",
            "is_implicit": False,
            "scenario_tier": "high_risk",
            "safety_flag": "high",
            "support_need": "safety_support",
            "evidence": ["cannot promise I will be safe", "alone tonight"],
            "crisis_signals": ["stated inability to stay safe alone"],
        },
    },
    {
        "name": "advice request",
        "user_message": "How do I talk to my roommate without making this worse?",
        "understanding": {
            "primary_emotion": "anxiety",
            "secondary_emotions": ["anger"],
            "emotion_intensity": 0.58,
            "intent": "advice_request",
            "is_implicit": False,
            "scenario_tier": "common",
            "safety_flag": "none",
            "support_need": "problem_solving",
            "evidence": ["talk to my roommate", "without making this worse"],
            "crisis_signals": [],
        },
    },
]


def main():
    for example in EXAMPLES:
        plan = generate_support_plan(
            example["understanding"],
            user_message=example["user_message"],
        )
        print(f"\n## {example['name']}")
        print(json.dumps(plan.model_dump(), indent=2))


if __name__ == "__main__":
    main()
