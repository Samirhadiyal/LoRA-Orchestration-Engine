# database/engine.py
import os

from dotenv import load_dotenv
from sqlmodel import create_engine

# Load environment variables from the .env file.
load_dotenv()

# Provide a fallback localhost URL if DATABASE_URL is missing from .env
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://postgres:password@localhost:5432/neuromesh_db"
)

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set in the environment variables.")

# create_engine establishes the connection pool.
engine = create_engine(DATABASE_URL, echo=True)