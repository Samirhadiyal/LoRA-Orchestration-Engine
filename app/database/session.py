# app/database/session.py
from sqlmodel import Session
from app.database.engine import engine

def get_session():
    """
    Dependency function that yields a database session for FastAPI routes.
    Automatically closes the session after the request is completed.
    """
    with Session(engine) as session:
        yield session