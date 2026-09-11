from typing import List

from pydantic import BaseModel, Field


class ATSReport(BaseModel):
    ats_score: float

    strengths: List[str] = Field(default_factory=list)
    issues: List[str] = Field(default_factory=list)

    matched_keywords: List[str] = Field(default_factory=list)
    missing_keywords: List[str] = Field(default_factory=list)

    recommendations: List[str] = Field(default_factory=list)
