
from fastapi import FastAPI
from app.api.routes import router as api_router  # ✅ Correct import

app = FastAPI(title="LLM Research Assistant", version="1.0")

# ✅ Register routes under the /api prefix
app.include_router(api_router, prefix="/api")

@app.get("/")
def read_root():
    return {"message": "LLM Research Assistant API is running"}
