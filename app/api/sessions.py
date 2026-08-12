import uuid
from typing import Annotated

from app.models.session import ChatSession
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session  # Import Session from sqlmodel

from app.database.session import get_db
from app.memory.redis_client import session_manager
router = APIRouter(prefix="/sessions", tags=["Sessions"])

@router.post("/")
async def create_session(user_id: str, db: Annotated[Session, Depends(get_db)]):
    """
    Creates a new chat session in Postgres and initializes Redis memory.
    """
    session_id = str(uuid.uuid4())
    
    try:
        # 1. Save session metadata permanently in PostgreSQL
        new_db_session = ChatSession(
            id=session_id,
            user_id=user_id
        )
        db.add(new_db_session)
        db.commit()
        db.refresh(new_db_session)

        # 2. Initialize temporary chat history in Redis
        await session_manager.add_message(
            session_id=session_id, 
            role="system", 
            content="You are NeuroMesh, an Adaptive AI Orchestration Platform."
        )

        return {
            "message": "Session successfully created in Postgres and Redis", 
            "session_id": session_id,
            "user_id": user_id,
            "created_at": new_db_session.created_at
        }
    except Exception as e:  # noqa: BLE001
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create session: {e!s}")
