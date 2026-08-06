# database/engine.py
import os
from sqlmodel import create_engine
from dotenv import load_dotenv

# Load environment variables from the .env file.
# We do this here so the engine always has the URL, even if run outside FastAPI (like in Alembic).
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set in the environment variables.")

# create_engine establishes the connection pool.
# echo=True prints raw SQL to the terminal. Useful for debugging, but should be False in production.
engine = create_engine(DATABASE_URL, echo=True)