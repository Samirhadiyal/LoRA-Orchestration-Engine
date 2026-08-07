# app/ingestion/chunker.py
import uuid
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentChunker:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Initializes the token-aware chunker using tiktoken (cl100k_base).
        This ensures we measure chunk size by AI tokens, not just raw characters.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def chunk_document(
        self, 
        pages_data: List[Dict[str, Any]], 
        document_id: uuid.UUID, 
        filename: str
    ) -> List[Dict[str, Any]]:
        """
        Takes parsed pages data and splits them into token-aware chunks.
        Attaches document-level and chunk-level metadata.
        """
        final_chunks = []
        global_chunk_index = 0

        for page in pages_data:
            text = page.get("text", "")
            page_number = page.get("page_number")

            if not text or not text.strip():
                continue

            # Split the page text into smaller token-aware pieces
            text_chunks = self.splitter.split_text(text)

            for chunk_text in text_chunks:
                # Structure matches the DocumentChunk database model
                chunk_data = {
                    "document_id": document_id,
                    "chunk_index": global_chunk_index,
                    "text": chunk_text,
                    "page_number": page_number,
                    "chunk_metadata": {
                        "source_filename": filename,
                        "page_number": page_number,
                    }
                }
                final_chunks.append(chunk_data)
                global_chunk_index += 1

        return final_chunks