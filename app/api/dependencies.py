from pydantic import BaseModel
from typing import List, Tuple, Optional

class QueryRequest(BaseModel):
    query: str
    history: Optional[List[Tuple[str, str]]] = []

class FeedbackRequest(BaseModel):
    query: str
    response: str
    user_feedback: str