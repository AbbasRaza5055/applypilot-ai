from typing import List, Optional

from pydantic import BaseModel, Field


class ApplicationPackage(BaseModel):
    cover_letter: Optional[str] = None
    sop: Optional[str] = None

    application_answers: List[str] = Field(default_factory=list)
    required_documents: List[str] = Field(default_factory=list)
    checklist: List[str] = Field(default_factory=list)
