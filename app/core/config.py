# app/core/config.py
import os

from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_NAME: str = "NeuroMesh API"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api/v1"
    
    # Database connections
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/lora_engine")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    # ... existing settings ...
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    
settings = Settings()