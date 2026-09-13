from typing import List
from pydantic import BaseModel, Field


class ResumeOptimizationResult(BaseModel):
    optimized_resume: str

    changed_sections: List[str] = Field(default_factory=list)
    added_keywords: List[str] = Field(default_factory=list)
    removed_or_weak_content: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)