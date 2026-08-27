# app/api/documents.py
import hashlib
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlmodel import Session, select

from app.database.models.document import Document, DocumentChunk, ProcessingStatus
from app.database.session import get_session
from app.ingestion.chunker import DocumentChunker
from app.ingestion.parsers import DocumentParser

router = APIRouter(prefix="/documents", tags=["Documents"])
chunker = DocumentChunker()

@router.post("/upload")
async def upload_document(
    file: Annotated[UploadFile, File(...)],
    db: Annotated[Session, Depends(get_session)],
) -> dict[str, Any]:
    """
    Uploads a document, parses it, chunks it, and saves metadata to the database.
    """
    # 1. Read file and calculate checksum to prevent duplicates
    file_bytes = await file.read()
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    
    existing_doc = db.exec(select(Document).where(Document.checksum == file_hash)).first()
    if existing_doc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Document '{file.filename}' already exists in the system."
        )

    # 2. Create the Document record in PostgreSQL
    db_document = Document(
        filename=file.filename,
        mime_type=file.content_type or "application/octet-stream",
        file_size_bytes=len(file_bytes),
        file_path=f"/temp/{file.filename}", # Placeholder for actual cloud storage path
        checksum=file_hash,
        status=ProcessingStatus.PROCESSING
    )
    db.add(db_document)
    db.commit()
    db.refresh(db_document)

    try:
        # 3. Parse the document
        parsed_pages = DocumentParser.parse(file_bytes=file_bytes, mime_type=db_document.mime_type)
        
        # 4. Chunk the document
        chunks_data = chunker.chunk_document(
            pages_data=parsed_pages, 
            document_id=UUID(str(db_document.id)),
            filename=db_document.filename
        )
        
        # 5. Save Chunks to PostgreSQL
        for chunk in chunks_data:
            db_chunk = DocumentChunk(
                document_id=chunk.document_id,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                page_number=chunk.page_number,
                chunk_metadata=chunk.chunk_metadata
            )
            db.add(db_chunk)
            
        # Update status to completed
        db_document.status = ProcessingStatus.COMPLETED
        db.commit()

        # NOTE: At this point, the chunks are ready to be passed to Engineer B's Embeddings module!
        from app.retrieval.indexer import index_chunks
        
        # Prepare chunks for Qdrant indexing
        qdrant_chunks = []
        for chunk in chunks_data:
            qdrant_chunks.append({
                "doc_id": str(db_document.id),
                "text": chunk.text,
                "metadata": chunk.chunk_metadata
            })
            
        index_chunks(qdrant_chunks)

        return {
            "status": "success",
            "document_id": db_document.id,
            "filename": db_document.filename,
            "total_chunks_created": len(chunks_data)
        }

    except Exception as e:  # noqa: BLE001
        db_document.status = ProcessingStatus.FAILED
        db_document.error_message = str(e)
        db.commit()
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e!s}")
