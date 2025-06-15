from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str

class FeedbackRequest(BaseModel):
    query: str
    response: str
    user_feedback: str