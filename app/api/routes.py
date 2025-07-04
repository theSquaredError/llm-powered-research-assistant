from fastapi import APIRouter
# from app.core.inference import answer_query
from app.core.inference import generate_answer_stream
from fastapi.responses import StreamingResponse
from app.core.feedback import store_feedback
from app.api.dependencies import QueryRequest, FeedbackRequest

router = APIRouter()

@router.post("/ask")
async def ask(payload: QueryRequest):
    def event_stream():
        try:
            for chunk in generate_answer_stream(payload.query, payload.history):
                yield chunk
        except Exception as e:
            yield f"Error: {type(e).__name__} -  {str(e)}"
    
    # response = generate_answer_stream(payload.query)
    return StreamingResponse(event_stream(), media_type="text/plain")

@router.post("/feedback")
async def feedback(payload: FeedbackRequest):
    store_feedback(payload.dict())
    return {"status": "Feedback recorded"}