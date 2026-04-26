# LLM #1 — Input Processing + Emotion & Intent Understanding

This document describes the structured Emotion + Intent Understanding layer
that sits between the existing GoEmotions emotion classifier and the response
generator.

---

## 1. What this module does

```
User Input → Input Processing → Emotion + Intent Understanding (LLM #1) → LLM #2 → Final Reply
                                ↑↑↑ this module ↑↑↑
```

It consumes the GoEmotions classifier output and produces a richer
**`UnderstandingState`** object that downstream code can use:

- The response generator (LLM #2) reads it to choose the appropriate
  support strategy and tone.
- The `/chat` endpoint exposes it as an additional `understanding` field
  in the response JSON.
- Future evaluation work can slice metrics by the `scenario_tier` field.

The module is purely additive: no existing code or behaviour is removed.
The team's `app/safety.py` crisis detector keeps full authority over the
top-level `safety.crisis_detected` boolean. The Understanding layer is
allowed only to **escalate** safety, never to lower it.

> **Note on RAG:** the original proposal showed a RAG retrieval step
> between LLM #1 and LLM #2. That step was dropped because the empathy
> LoRA-adapted Qwen2.5 model already absorbs the relevant empathy
> knowledge during fine-tuning. The output schema therefore does not
> include any retrieval-related field.

---

## 2. File map

```
app/
├── main.py                    (existing — additive edits only)
├── prompting.py               (existing — additive edits only)
├── schemas.py                 (existing — additive edits only)
├── safety.py                  (existing — untouched)
├── session.py                 (existing — untouched)
├── models/emotion.py          (existing GoEmotions classifier — untouched)
├── models/generator.py        (existing Qwen2.5 + LoRA generator — untouched)
└── understanding/             ★ NEW MODULE ★
    ├── __init__.py            public exports: UnderstandingAnalyzer, UnderstandingState
    ├── analyzer.py            entry point: analyze(message, history, emotions)
    ├── schemas.py             Pydantic schemas for the structured output
    │
    ├── input_processing.py    facade for the Input Processing stage:
    ├── pii_redactor.py          • email/phone/URL → [PLACEHOLDER]
    ├── lexical_features.py      • LIWC-style word counts + valence/arousal
    ├── safety_fusion.py         • combines safety.py with extended lexicon
    │
    ├── intent_classifier.py   12-category intent classifier (rule-based)
    ├── intensity.py           emotion intensity ∈ [0, 1]
    ├── implicit_detector.py   explicit vs implicit + suppressed-feeling detector
    ├── scenario_tier.py       common / subtle / high_risk
    ├── support_need.py        directive for the response generator
    ├── personality_tracker.py optional Big-Five tendency, confidence-gated
    └── fusion.py              combines all signals into UnderstandingState

data/
├── lexicons/
│   ├── crisis_keywords.txt           extended crisis lexicon (passive ideation phrases)
│   ├── nrc_emolex_starter.csv        NRC EmoLex starter dictionary (free, academic-license)
│   └── nrc_vad_starter.csv           NRC VAD starter dictionary (free, academic-license)
└── eval_samples/
    ├── common.jsonl                  clear-emotion samples used by unit tests
    ├── subtle.jsonl                  implicit-emotion samples
    └── high_risk.jsonl               crisis samples + a metaphorical false-positive case

scripts/
└── demo_understanding.py             standalone demo, no ML model load required

tests/
└── test_understanding.py             10 unit tests
```

---

## 3. Output contract — `UnderstandingState`

Every `/chat` request now returns an additional `understanding` field on
the response JSON, mirroring `app/understanding/schemas.py::UnderstandingState`.

| Field | Type | Description |
|---|---|---|
| `primary_emotion` | string | GoEmotions label of the dominant emotion |
| `secondary_emotions` | list[string] | up to 2 additional GoEmotions labels |
| `emotion_intensity` | float in [0, 1] | strength of the emotion (different from classifier confidence) |
| `valence` | float in [-1, +1] | pleasantness (from NRC VAD lexicon) |
| `arousal` | float in [0, 1] | activation level (from NRC VAD lexicon) |
| `intent` | string | one of 12 categories: `emotional_support`, `advice_request`, `validation_seeking`, `venting`, `crisis_or_safety`, `small_talk`, `information_request`, `planning_help`, `relationship_support`, `work_or_school_stress`, `health_or_body_concern`, `other` |
| `is_implicit` | bool | true iff the user did not name a feeling explicitly |
| `scenario_tier` | string | `common` / `subtle` / `high_risk` |
| `safety_flag` | string | `none` / `low` / `medium` / `high` |
| `crisis_signals` | list[string] | human-readable explanations of the safety flag |
| `support_need` | string | strategy directive for the response generator (e.g. `validation_and_gentle_exploration`, `grounding`, `safety_support`, `problem_solving`) |
| `evidence` | list[string] | 1–3 short phrases explaining the decision (for audit / debugging) |
| `personality` | object \| null | optional Big-Five snapshot, only emitted when confidence ≥ 0.4 |
| `personality_confidence` | float in [0, 1] | grows with accumulated user text |
| `pii_redacted_text` | string | log-safe version of the input (email/phone/URL replaced) |
| `layer_outputs` | object | raw outputs of each internal layer (debug only) |

---

## 4. Pipeline behaviour per `/chat` request

```
request.message  ──┐
                   │
session history    │   ┌────────────────────────────────────────────────┐
                   ├──>│ build_classifier_input  (existing helper)     │
                   │   └──────────────────┬─────────────────────────────┘
                   │                      ▼
                   │   ┌────────────────────────────────────────────────┐
                   │   │ EmotionClassifier.predict  (GoEmotions)        │
                   │   └──────────────────┬─────────────────────────────┘
                   │                      ▼ emotions = {primary, ...}
                   │   ┌────────────────────────────────────────────────┐
                   ├──>│ UnderstandingAnalyzer.analyze   (★ this module)│
                   │   │  Stage 1: Input Processing                     │
                   │   │    normalize → PII redact → lexical features   │
                   │   │    → safety prefilter                          │
                   │   │  Stage 2: Personality (optional, multi-turn)   │
                   │   │  Stage 3: Fusion → UnderstandingState          │
                   │   └──────────────────┬─────────────────────────────┘
                   │                      ▼ understanding payload
                   │   ┌────────────────────────────────────────────────┐
                   │   │ if understanding.safety_flag in {medium,high}: │
                   │   │     escalate crisis_detected = True            │
                   │   │ if crisis_detected:                            │
                   │   │     reply = crisis_reply()                     │
                   │   │ else:                                          │
                   │   │     messages = build_generation_messages_      │
                   │   │                with_understanding(...)         │
                   │   │     reply = generator.generate(messages)       │
                   │   └──────────────────┬─────────────────────────────┘
                   ▼                      ▼
              session_store           ChatResponse(emotions, understanding, ...)
```

**Key safety design rule.** Safety is one-way: any layer can escalate the
flag, but no layer can lower another's escalation. This protects against
a classifier that confidently mis-labels a crisis message as `neutral`.

---

## 5. Decisions on external resources

| Resource | Decision | Reason |
|---|---|---|
| **LIWC-22** ([liwc.app](https://www.liwc.app/)) | Used as a design concept only; not a runtime dependency. | LIWC-22 is closed-source/paid. Free academic equivalents (NRC EmoLex + NRC VAD) cover the same word-category counting use case and are bundled in `data/lexicons/`. |
| **HuggingFace `AliiaR/DialoGPT-medium-empathetic-dialogues`** | Not used in this layer. | It is a generative response model, not an emotion classifier. It is a possible alternative for the LLM #2 generator path, not for LLM #1. |
| **Kaggle EmpatheticDialogues** ([dataset](https://www.kaggle.com/datasets/atharvjairath/empathetic-dialogues-facebook-ai)) | Not used in this layer right now. | This module ships ~20 hand-curated samples in `data/eval_samples/` for unit tests. The Kaggle dataset is only relevant for any future fine-tuning or large-scale evaluation work. |

### Lexicon upgrade path

The shipped lexicons (`nrc_emolex_starter.csv`, `nrc_vad_starter.csv`) are
small starter dictionaries (~70 words each) so the code runs out of the
box with no downloads. For production-grade quality, download the full
free lexicons from [saifmohammad.com/WebPages/lexicons.html](https://saifmohammad.com/WebPages/lexicons.html)
and place them at:

- `data/lexicons/nrc_emolex_full.csv`
- `data/lexicons/nrc_vad_full.csv`

The loader auto-detects the full versions and uses them when present.

---

## 6. Personality tracker (optional, off by default in short chats)

`personality_tracker.py` accumulates the user's text across the session
and computes a Big-Five (OCEAN) tendency estimate. This is **not a
psychological diagnosis** and is intentionally conservative:

| Total user words | Confidence (capped at 0.65) |
|---:|---|
| < 150 | < 0.20 |
| 150–400 | 0.20–0.35 |
| 400–800 | 0.35–0.50 |
| 800–1500 | 0.50–0.65 |
| 1500+ | 0.65 (hard cap) |

If confidence is below `0.4`, the tracker returns `None` and
downstream consumers ignore personality entirely. This means short
demos publish no personality output, which is the safe default.

---

## 7. Running the module

### Install minimal dependency

The Understanding layer itself only needs Pydantic:

```bash
pip install "pydantic>=2"
```

(The full backend has heavier dependencies like Unsloth, PyTorch + CUDA,
etc. — those are required only for the response generator path.)

### Run the unit tests

```bash
python -m unittest tests.test_understanding tests.test_session tests.test_prompting
```

Expected output: `Ran 14 tests in 0.0XXs ... OK`.

### Run the standalone demo (no ML models loaded)

```bash
python -m scripts.demo_understanding
```

The demo synthesises GoEmotions-style inputs and walks through 8
representative messages (clear emotion, implicit/subtle, crisis,
metaphorical false-positive). Useful for sanity-checking the
analyzer's output without standing up the full backend.

---

## 8. Suppressed-feeling detection

A small but meaningful feature: when the input contains a "should be /
supposed to feel" pattern, the implicit detector overrides any
explicit-emotion-word match and marks the case as implicit. Example:

```
Input:  "My boss said the project went well. I should be happy."
Output: scenario_tier="subtle", is_implicit=True,
        support_need="validation_and_gentle_exploration"
```

This catches a real linguistic phenomenon (performing a feeling vs
having one) that simple keyword classification misses.

---

## 9. Concrete examples (from the demo)

| Input | Behaviour without this module | Behaviour with this module |
|---|---|---|
| `"Sometimes I think everyone would be better off without me."` | `safety.crisis_detected=False` (passive ideation phrase not in the original 10-pattern detector) | `safety_flag="high"`, routes to safety reply |
| `"Yeah... I guess. Whatever."` | classifier returns `neutral`, generic reply | `is_implicit=True`, `scenario_tier="subtle"`, `support_need="validation_and_gentle_exploration"` |
| `"I'm so stressed. What should I do?"` | classifier returns `nervousness`, no intent signal | adds `intent="advice_request"`, `support_need="problem_solving"` |
| `"This homework is killing me, lol."` | classifier may or may not flag | `safety_flag="none"` (false-positive resistance verified by test) |

---

## 10. Test coverage

`tests/test_understanding.py` contains 10 tests:

1. Schema validity on `common.jsonl` samples
2. Schema validity + `is_implicit=True` on `subtle.jsonl` samples
3. Safety escalation on `high_risk.jsonl` samples
4. Metaphorical "killing me, lol" not escalated
5. PII redaction (email / phone / URL)
6. Intent classification — advice_request
7. Intent classification — small_talk
8. Intent classification — crisis_or_safety on explicit ideation
9. Personality tracker returns `None` for short messages
10. Understanding payload reaches the generator prompt
