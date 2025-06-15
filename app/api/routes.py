from fastapi import APIRouter
from app.core.inference import answer_query
from app.core.feedback import store_feedback
from app.api.dependencies import QueryRequest, FeedbackRequest

router = APIRouter()

@router.post("/ask")
async def ask(payload: QueryRequest):
    response = answer_query(payload.query)
    return {"response": response}

@router.post("/feedback")
async def feedback(payload: FeedbackRequest):
    store_feedback(payload.dict())
    return {"status": "Feedback recorded"}