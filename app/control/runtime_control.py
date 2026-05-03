from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.planning import generate_support_plan
from evaluation.refine import refine_response
from evaluation.validate import validate_response_structured


@dataclass
class RuntimeControlResult:
    final_reply: str
    anchor_reply: str
    support_plan: dict[str, Any] | None
    validation: dict[str, Any]
    refined_validation: dict[str, Any]
    refined: bool
    failure_types: list[str]
    refinement_reason: str


def _as_dict(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return None


def apply_runtime_control(
    *,
    user_message: str,
    anchor_reply: str,
    scenario_type: str | None = None,
    understanding: Any = None,
) -> RuntimeControlResult:
    understanding_dict = _as_dict(understanding)
    support_plan_model = (
        generate_support_plan(understanding_dict, user_message=user_message)
        if understanding_dict
        else None
    )
    support_plan = support_plan_model.model_dump() if support_plan_model else None

    validation = validate_response_structured(
        user_message=user_message,
        response=anchor_reply,
        scenario_type=scenario_type,
        understanding=understanding_dict,
    )

    refined = not validation["passed"]
    if refined:
        final_reply = refine_response(
            user_message=user_message,
            original_response=anchor_reply,
            validation_result=validation,
            support_plan=support_plan,
            understanding=understanding_dict,
            scenario_type=scenario_type,
        )
        refinement_reason = ", ".join(validation["failure_types"])
    else:
        final_reply = anchor_reply
        refinement_reason = ""

    refined_validation = validate_response_structured(
        user_message=user_message,
        response=final_reply,
        scenario_type=scenario_type,
        understanding=understanding_dict,
    )

    return RuntimeControlResult(
        final_reply=final_reply,
        anchor_reply=anchor_reply,
        support_plan=support_plan,
        validation=validation,
        refined_validation=refined_validation,
        refined=refined,
        failure_types=validation["failure_types"],
        refinement_reason=refinement_reason,
    )
