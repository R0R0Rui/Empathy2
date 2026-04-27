"""
analyzer.py
===========
The PUBLIC entry point of the Understanding layer.

Used in app/main.py like:

    from app.understanding import UnderstandingAnalyzer

    analyzer = UnderstandingAnalyzer()  # constructed once at startup
    state = analyzer.analyze(
        user_message=request.message,
        history=history,                   # team's session history list
        emotions=emotions,                 # the GoEmotions classifier output
    )

The analyzer is STATELESS. It does not own any session state. It receives
the conversation history per-call and computes everything fresh, which
keeps it safe to share across threads/requests.
"""

from __future__ import annotations
from typing import Sequence

from app.session import Turn

from .schemas import UnderstandingState
from .input_processing import process_input
from .personality_tracker import estimate_personality
from .fusion import fuse_into_understanding_state


class UnderstandingAnalyzer:
    """
    Orchestrates the Understanding layer.

    The pipeline is two clearly named stages:
      1. Input Processing   (input_processing.process_input)
      2. Emotion + Intent Understanding fusion  (fusion.fuse_into_understanding_state)

    Designed to be called inline in /chat. The whole pipeline is
    <50ms even on CPU because the heaviest model (GoEmotions) was
    already run by the team's classifier upstream.
    """

    def analyze(
        self,
        user_message: str,
        history: Sequence[Turn],
        emotions: dict,
    ) -> UnderstandingState:
        # === STAGE 1: Input Processing ===
        # Normalize text, redact PII for logs, extract lexical features,
        # run safety prefilter. See input_processing.py for the breakdown.
        processed = process_input(user_message)

        # === STAGE 2: Personality (optional, multi-turn) ===
        # Accumulate ALL prior user turns + this one for the Big-Five tracker.
        user_text_concat = "\n".join(
            t["content"] for t in history if t["role"] == "user"
        )
        if user_text_concat:
            user_text_concat += "\n"
        user_text_concat += user_message
        bf, bf_conf = estimate_personality(user_text_concat)

        # === STAGE 3: Emotion + Intent Understanding fusion ===
        return fuse_into_understanding_state(
            user_text=user_message,
            pii_redacted_text=processed.pii_redacted_text,
            classifier_emotions=emotions,
            lex=processed.lexical,
            safety=processed.safety,
            personality=bf,
            personality_confidence=bf_conf,
        )
