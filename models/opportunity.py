from typing import List, Optional

from pydantic import BaseModel, Field


class Opportunity(BaseModel):
    id: str
    title: str
    organization: str
    type: str
    description: str

    requirements: List[str] = Field(default_factory=list)

    deadline: Optional[str] = None
    location: Optional[str] = None

    url: str
    source: Optional[str] = None
