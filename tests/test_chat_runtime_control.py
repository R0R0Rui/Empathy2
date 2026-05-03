import unittest
import sys
import types

from fastapi.testclient import TestClient

if "unsloth" not in sys.modules:
    fake_unsloth = types.ModuleType("unsloth")
    fake_unsloth.FastLanguageModel = types.SimpleNamespace()
    sys.modules["unsloth"] = fake_unsloth

import app.main as main


def _fake_emotions(primary="fear", secondary="sadness"):
    return {
        "primary": {"label": primary, "confidence": 0.7},
        "secondary": {"label": secondary, "confidence": 0.2},
        "tertiary": {"label": "neutral", "confidence": 0.1},
    }


class FakeClassifier:
    def __init__(self, emotions=None):
        self.emotions = emotions or _fake_emotions()

    def predict(self, _text):
        return self.emotions


class FakeGenerator:
    def __init__(self, reply):
        self.reply = reply

    def generate(self, _messages):
        return self.reply


class FakeUnderstandingState:
    def __init__(self, **overrides):
        self.payload = {
            "primary_emotion": "fear",
            "secondary_emotions": ["sadness"],
            "emotion_intensity": 0.6,
            "valence": -0.4,
            "arousal": 0.6,
            "intent": "emotional_support",
            "is_implicit": False,
            "scenario_tier": "common",
            "safety_flag": "none",
            "crisis_signals": [],
            "support_need": "validation",
            "evidence": ["test evidence"],
            "personality": None,
            "personality_confidence": 0.0,
            "pii_redacted_text": "redacted",
            "layer_outputs": {},
        }
        self.payload.update(overrides)

    def model_dump(self):
        return self.payload


class FakeAnalyzer:
    def __init__(self, state):
        self.state = state

    def analyze(self, **_kwargs):
        return self.state


class ChatRuntimeControlTests(unittest.TestCase):
    def setUp(self):
        self.old_classifier = getattr(main, "emotion_classifier", None)
        self.old_generator = getattr(main, "generator", None)
        self.old_analyzer = main.understanding_analyzer
        self.client = TestClient(main.app)

    def tearDown(self):
        main.emotion_classifier = self.old_classifier
        main.generator = self.old_generator
        main.understanding_analyzer = self.old_analyzer

    def test_chat_returns_support_plan_and_validation_metadata(self):
        main.emotion_classifier = FakeClassifier()
        main.generator = FakeGenerator("You should prepare and do your best.")
        main.understanding_analyzer = FakeAnalyzer(FakeUnderstandingState())

        response = self.client.post(
            "/chat",
            json={
                "message": "I got the internship I wanted, but now I am nervous they made a mistake choosing me."
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("support_plan", body)
        self.assertIn("validation", body)
        self.assertIn("anchor_reply", body)
        self.assertIn("refined", body)
        self.assertEqual(body["anchor_reply"], "You should prepare and do your best.")

    def test_chat_reply_is_refined_when_anchor_fails(self):
        main.emotion_classifier = FakeClassifier()
        main.generator = FakeGenerator("Oh wow that is so exciting! I'm sure you did great.")
        main.understanding_analyzer = FakeAnalyzer(FakeUnderstandingState())

        response = self.client.post(
            "/chat",
            json={
                "message": "I got the internship I wanted, but now I am nervous they made a mistake choosing me."
            },
        )

        body = response.json()
        self.assertTrue(body["refined"])
        self.assertNotEqual(body["reply"], body["anchor_reply"])
        self.assertIn("avoids_core_issue", body["failure_types"])
        self.assertIn("made a mistake", body["reply"])
        self.assertIn("proud", body["reply"])

    def test_final_reply_has_no_meta_instruction_leakage(self):
        main.emotion_classifier = FakeClassifier()
        main.generator = FakeGenerator("You should prepare and do your best.")
        main.understanding_analyzer = FakeAnalyzer(FakeUnderstandingState())

        response = self.client.post(
            "/chat",
            json={
                "message": "I got the internship I wanted, but now I am nervous they made a mistake choosing me."
            },
        )

        reply = response.json()["reply"].lower()
        banned = [
            "what stands out is",
            "for this response",
            "the priority is",
            "support plan",
            "failure type",
            "validate the user's reaction",
        ]
        self.assertFalse(any(phrase in reply for phrase in banned))

    def test_high_risk_cases_include_safety_language(self):
        main.emotion_classifier = FakeClassifier(_fake_emotions("sadness", "fear"))
        main.generator = FakeGenerator("That sounds hard.")
        main.understanding_analyzer = FakeAnalyzer(
            FakeUnderstandingState(
                intent="crisis_or_safety",
                scenario_tier="high_risk",
                safety_flag="high",
                support_need="safety_support",
                crisis_signals=["explicit crisis language"],
            )
        )

        response = self.client.post(
            "/chat",
            json={"message": "I do not see the point in being here anymore."},
        )

        body = response.json()
        self.assertIn("988", body["reply"])
        self.assertIn("emergency", body["reply"])
        self.assertTrue(body["safety"]["crisis_detected"])


if __name__ == "__main__":
    unittest.main()
