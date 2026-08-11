# app/api/orchestrate.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.orchestration_service import OrchestrationService

router = APIRouter(tags=["Orchestration"])

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None

@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Core execution endpoint. Accepts a user message, passes it to the 
    orchestration service, and returns the execution results.
    """
    try:
        response = await OrchestrationService.process_chat(
            user_query=request.message,
            session_id=request.session_id
        )
        return response
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))
