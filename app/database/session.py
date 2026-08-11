from sqlmodel import Session

from app.database.engine import engine


def get_session():
    """FastAPI dependency to yield a database session."""
    with Session(engine) as session:
        yield session

def get_db():
    """Alias for backwards compatibility with other routers."""
    with Session(engine) as session:
        yield session