from app.planning import generate_support_plan


def test_subtle_self_dismissal_plan():
    plan = generate_support_plan(
        {
            "primary_emotion": "sadness",
            "secondary_emotions": ["loneliness"],
            "emotion_intensity": 0.6,
            "intent": "venting",
            "is_implicit": True,
            "scenario_tier": "subtle",
            "safety_flag": "none",
            "support_need": "validation_and_gentle_exploration",
            "evidence": ["being dramatic", "felt ignored"],
            "crisis_signals": [],
        },
        user_message="I am probably being dramatic, but I felt ignored.",
    )

    assert plan.support_goal == "reduce self-dismissal and validate the underlying feeling"
    assert "reflect implicit feeling" in plan.response_acts
    assert "validate without overstatement" in plan.response_acts
    assert "do not agree with negative self-framing" in plan.constraints


def test_high_risk_safety_support_plan():
    plan = generate_support_plan(
        {
            "primary_emotion": "fear",
            "secondary_emotions": ["sadness"],
            "emotion_intensity": 0.92,
            "intent": "crisis_or_safety",
            "is_implicit": False,
            "scenario_tier": "high_risk",
            "safety_flag": "high",
            "support_need": "safety_support",
            "evidence": ["cannot stay safe"],
            "crisis_signals": ["cannot promise safety"],
        }
    )

    assert plan.support_goal == "support immediate safety and connection to appropriate help"
    assert "provide safety-aware support" in plan.response_acts
    assert plan.safety_notes
    assert "add concrete safety support" in plan.repair_priorities


def test_advice_request_plan_validates_before_problem_solving():
    plan = generate_support_plan(
        {
            "primary_emotion": "anxiety",
            "secondary_emotions": [],
            "emotion_intensity": 0.5,
            "intent": "advice_request",
            "is_implicit": False,
            "scenario_tier": "common",
            "safety_flag": "none",
            "support_need": "problem_solving",
            "evidence": ["how do I talk to my roommate"],
            "crisis_signals": [],
        }
    )

    assert "validate emotion before problem-solving" in plan.response_acts
    assert "lead with emotional validation before advice" in plan.constraints
    assert "jumping straight into instructions" in plan.avoid


def test_generic_common_emotion_plan_has_base_constraints():
    plan = generate_support_plan(
        {
            "primary_emotion": "joy",
            "secondary_emotions": [],
            "emotion_intensity": 0.4,
            "intent": "reflection",
            "is_implicit": False,
            "scenario_tier": "common",
            "safety_flag": "none",
            "support_need": "encouragement",
            "evidence": [],
            "crisis_signals": [],
        }
    )

    assert plan.support_goal == "recognize effort and offer grounded encouragement"
    assert "do not claim personal experience" in plan.constraints
    assert "do not pretend to be human" in plan.constraints
    assert "do not give generic reassurance only" in plan.constraints


def test_implicit_suppressed_feeling_plan():
    plan = generate_support_plan(
        {
            "primary_emotion": "anxiety",
            "secondary_emotions": ["sadness"],
            "emotion_intensity": 0.7,
            "intent": "emotional_support",
            "is_implicit": True,
            "scenario_tier": "subtle",
            "safety_flag": "none",
            "support_need": "validation_and_gentle_exploration",
            "evidence": ["nodded like it was fine"],
            "crisis_signals": [],
        },
        user_message="The deadline moved up. I nodded like it was fine.",
    )

    assert "reflect implicit feeling" in plan.response_acts
    assert "invite gentle exploration" in plan.response_acts
    assert "use tentative language for inferred feelings" in plan.constraints
