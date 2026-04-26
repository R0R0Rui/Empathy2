"""
lexical_features.py
===================
LIWC-style word counting from FREE, open-source lexicons.

WHY THIS LAYER EXISTS (vs just trusting GoEmotions + LLM):
  - GoEmotions is great at categorical labels but doesn't give us continuous
    valence/arousal values that the proposal references via BEAM.
  - Lexicon counts are interpretable: we can show the team and the professor
    "we found 3 sadness words and 1 anxiety word in this message", which
    makes the system explainable.
  - Crucially, lexicon features are HARD TO FOOL by short, implicit messages
    where GoEmotions softmaxes everything to 'neutral'.

WHY NOT LIWC.APP DIRECTLY:
  LIWC-22 is a paid, closed-source product. We can't ship it. The free
  equivalents we use:
    - NRC EmoLex (Mohammad & Turney 2013): word -> emotion category
    - NRC VAD lexicon (Mohammad 2018): word -> (valence, arousal, dominance)
  Both are free for academic use from saifmohammad.com.

LOCAL FILE LOCATIONS:
  data/lexicons/nrc_emolex_starter.csv  (shipped, small)
  data/lexicons/nrc_vad_starter.csv     (shipped, small)
  data/lexicons/nrc_emolex_full.csv     (optional, you download separately)
  data/lexicons/nrc_vad_full.csv        (optional, you download separately)

  If the *_full.csv version exists, we use it. Otherwise we fall back to
  the starter dictionary. So out-of-the-box the code runs; with a download
  it gets better.
"""

from __future__ import annotations
import csv
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple


# Resolve data path relative to repo root (matches team's config.py style).
_REPO_ROOT = Path(__file__).resolve().parents[2]
LEXICON_DIR = _REPO_ROOT / "data" / "lexicons"


@dataclass
class LexicalFeatures:
    """Structured output from the lexical pass."""
    emotion_word_counts: Dict[str, int] = field(default_factory=dict)
    valence_score: float = 0.0          # -1..+1
    arousal_score: float = 0.5          # 0..1
    explicit_emotion_words: List[str] = field(default_factory=list)
    total_word_count: int = 0


# ---------------------------------------------------------------------------
# Loaders (cached)
# ---------------------------------------------------------------------------
# A small inline fallback so the module always works even if files are missing.
_FALLBACK_EMOLEX: Dict[str, Dict[str, int]] = {
    "sad":      {"sadness": 1},
    "lonely":   {"sadness": 1},
    "tired":    {"sadness": 1},
    "exhausted":{"sadness": 1},
    "depressed":{"sadness": 1},
    "hopeless": {"sadness": 1},
    "happy":    {"joy": 1},
    "excited":  {"joy": 1},
    "grateful": {"joy": 1, "trust": 1},
    "proud":    {"joy": 1},
    "afraid":   {"fear": 1},
    "anxious":  {"fear": 1},
    "worried":  {"fear": 1},
    "scared":   {"fear": 1},
    "stressed": {"fear": 1},
    "nervous":  {"fear": 1},
    "angry":    {"anger": 1},
    "furious":  {"anger": 1},
    "annoyed":  {"anger": 1},
    "frustrated":{"anger": 1},
    "ashamed":  {"sadness": 1},
    "guilty":   {"sadness": 1},
    "embarrassed":{"sadness": 1},
    "calm":     {"trust": 1},
    "confused": {"surprise": 1},
}


_FALLBACK_VAD: Dict[str, Tuple[float, float]] = {
    # word: (valence, arousal); both in [0, 1]
    "sad":         (0.05, 0.40),
    "lonely":      (0.10, 0.45),
    "tired":       (0.30, 0.20),
    "exhausted":   (0.20, 0.20),
    "depressed":   (0.05, 0.30),
    "hopeless":    (0.05, 0.40),
    "happy":       (0.95, 0.65),
    "excited":     (0.90, 0.85),
    "grateful":    (0.90, 0.55),
    "proud":       (0.90, 0.65),
    "afraid":      (0.15, 0.85),
    "anxious":     (0.20, 0.80),
    "worried":     (0.25, 0.75),
    "scared":      (0.15, 0.85),
    "stressed":    (0.20, 0.85),
    "nervous":     (0.25, 0.75),
    "angry":       (0.15, 0.85),
    "furious":     (0.10, 0.90),
    "annoyed":     (0.30, 0.65),
    "frustrated":  (0.20, 0.70),
    "ashamed":     (0.10, 0.55),
    "guilty":      (0.15, 0.55),
    "embarrassed": (0.20, 0.65),
    "calm":        (0.80, 0.20),
}


# Words people use to NAME a feeling explicitly. Used for is_implicit detection.
EXPLICIT_FEELING_WORDS = {
    "feel", "feeling", "felt", "emotion", "emotional", "mood",
    "sad", "sadness", "happy", "happiness", "joy", "joyful",
    "anxious", "anxiety", "afraid", "fear", "scared", "scary",
    "angry", "anger", "mad", "furious", "annoyed", "frustrated",
    "lonely", "loneliness", "ashamed", "shame",  "guilty", "guilt",
    "embarrassed", "stressed", "stress", "exhausted", "tired",
    "depressed", "hopeless", "grateful", "thankful", "proud",
    "excited", "nervous", "worried", "worry",
}


@lru_cache(maxsize=1)
def _load_emolex() -> Dict[str, Dict[str, int]]:
    full = LEXICON_DIR / "nrc_emolex_full.csv"
    starter = LEXICON_DIR / "nrc_emolex_starter.csv"
    path = full if full.exists() else starter
    if not path.exists():
        return _FALLBACK_EMOLEX
    out: Dict[str, Dict[str, int]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            w = row["word"].strip().lower()
            try:
                score = int(row["score"])
            except (ValueError, KeyError):
                continue
            if score <= 0:
                continue
            out.setdefault(w, {})[row["emotion"].strip().lower()] = score
    return out or _FALLBACK_EMOLEX


@lru_cache(maxsize=1)
def _load_vad() -> Dict[str, Tuple[float, float]]:
    full = LEXICON_DIR / "nrc_vad_full.csv"
    starter = LEXICON_DIR / "nrc_vad_starter.csv"
    path = full if full.exists() else starter
    if not path.exists():
        return _FALLBACK_VAD
    out: Dict[str, Tuple[float, float]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                v = float(row["valence"])
                a = float(row["arousal"])
            except (ValueError, KeyError):
                continue
            out[row["word"].strip().lower()] = (v, a)
    return out or _FALLBACK_VAD


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[a-z']+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def extract_lexical_features(text: str) -> LexicalFeatures:
    """Run text through the lexicons and return aggregated features."""
    tokens = _tokenize(text)
    if not tokens:
        return LexicalFeatures()

    emolex = _load_emolex()
    vad = _load_vad()

    emotion_counts: Dict[str, int] = {}
    valences: List[float] = []
    arousals: List[float] = []
    explicit_words: List[str] = []

    for tok in tokens:
        if tok in EXPLICIT_FEELING_WORDS:
            explicit_words.append(tok)

        if tok in emolex:
            for emo, score in emolex[tok].items():
                if emo in {"positive", "negative"}:
                    continue
                emotion_counts[emo] = emotion_counts.get(emo, 0) + 1

        if tok in vad:
            v, a = vad[tok]
            valences.append(v)
            arousals.append(a)

    # Map [0,1] valence to [-1,+1]; arousal stays in [0,1].
    valence_score = (sum(valences) / len(valences) * 2 - 1) if valences else 0.0
    arousal_score = (sum(arousals) / len(arousals)) if arousals else 0.5

    return LexicalFeatures(
        emotion_word_counts=emotion_counts,
        valence_score=round(valence_score, 3),
        arousal_score=round(arousal_score, 3),
        explicit_emotion_words=explicit_words,
        total_word_count=len(tokens),
    )
