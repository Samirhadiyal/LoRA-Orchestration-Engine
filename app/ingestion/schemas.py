# app/ingestion/schemas.py
from typing import Optional
from uuid import UUID
from pydantic import BaseModel

class ParsedPage(BaseModel):
    """Schema for a single parsed page from a document."""
    text: str
    page_number: Optional[int] = None

class ChunkData(BaseModel):
    """Schema for a single token-aware chunk."""
    document_id: UUID
    chunk_index: int
    text: str
    page_number: Optional[int] = None
    chunk_metadata: dict