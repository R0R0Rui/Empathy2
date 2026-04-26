"""
scenario_tier.py
================
Maps each user message into one of three tiers shown in the proposal's
Evaluation slide:
    common    -> clear, named emotion
    subtle    -> emotion is implicit / context-dependent
    high_risk -> safety-sensitive (self-harm, crisis, severe distress)

Decision tree:
    if safety_flag in {"medium", "high"}: high_risk
    elif is_implicit                     : subtle
    else                                 : common

The tier is what the future evaluation step (LLM Judge + Human Check) uses
to slice metrics. Anchor's claim to be "better than mainstream AI" rests
mostly on subtle and high_risk tiers, so accurate tier assignment matters.
"""

from __future__ import annotations
from typing import Literal

Tier = Literal["common", "subtle", "high_risk"]


def assign_scenario_tier(
    safety_flag: str,
    is_implicit: bool,
) -> Tier:
    if safety_flag in {"medium", "high"}:
        return "high_risk"
    if is_implicit:
        return "subtle"
    return "common"
