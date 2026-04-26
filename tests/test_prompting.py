import unittest

from app.prompting import build_classifier_input, build_generation_messages


class PromptingTests(unittest.TestCase):
    def test_classifier_input_contains_history_and_current_blocks(self):
        history = [
            {"role": "user", "content": "I've been overwhelmed."},
            {"role": "assistant", "content": "That sounds exhausting."},
        ]
        result = build_classifier_input("Nothing helps.", history)
        self.assertIn("[HISTORY]", result)
        self.assertIn("user: I've been overwhelmed.", result)
        self.assertIn("[CURRENT]", result)
        self.assertIn("user: Nothing helps.", result)

    def test_generation_prompt_includes_emotion_and_current_message(self):
        emotions = {
            "primary": {"label": "neutral", "confidence": 0.41},
            "secondary": {"label": "curiosity", "confidence": 0.31},
            "tertiary": {"label": "confusion", "confidence": 0.13},
        }
        messages = build_generation_messages("How does CBT work?", [], emotions)
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("Keep responses 2-4 sentences", messages[0]["content"])
        self.assertIn("Detected emotion profile: neutral 0.41, curiosity 0.31, confusion 0.13", messages[1]["content"])
        self.assertIn("User message:\n<user>How does CBT work?</user>", messages[1]["content"])
        self.assertIn("Respond with emotional validation first", messages[1]["content"])


if __name__ == "__main__":
    unittest.main()
