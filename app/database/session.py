# database/session.py
from sqlmodel import Session
from database.engine import engine

def get_session():
    """
    Dependency generator for FastAPI.
    Yields a database session and ensures it is properly closed after the request finishes.
    """
    with Session(engine) as session:
        yield session