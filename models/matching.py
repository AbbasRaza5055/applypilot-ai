from typing import List

from pydantic import BaseModel, Field


class MatchResult(BaseModel):
    opportunity_id: str
    match_score: float
    eligible: bool

    strengths: List[str] = Field(default_factory=list)
    gaps: List[str] = Field(default_factory=list)
    reasons: List[str] = Field(default_factory=list)
