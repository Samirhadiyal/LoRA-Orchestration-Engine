# app/database/session.py
from typing import Generator
from sqlmodel import Session
from app.database.engine import engine

def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that provides a transactional database session per request
    and automatically closes it when the request is finished.
    """
    with Session(engine) as session:
        yield session