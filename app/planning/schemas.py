from pydantic import BaseModel, Field


class SupportPlan(BaseModel):
    support_goal: str
    response_acts: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)
    safety_notes: list[str] = Field(default_factory=list)
    repair_priorities: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
