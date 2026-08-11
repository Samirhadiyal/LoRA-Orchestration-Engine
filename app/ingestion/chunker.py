# app/ingestion/chunker.py

import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingestion.schemas import ChunkData, ParsedPage


class DocumentChunker:
    """
    Splits parsed document pages into token-aware chunks.

    Uses OpenAI's cl100k_base tokenizer so chunk sizes are measured
    in tokens rather than characters.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = (
            RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name="cl100k_base",
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=[
                    "\n\n",
                    "\n",
                    ". ",
                    " ",
                    "",
                ],
            )
        )

    def chunk_document(
        self,
        pages_data: list[ParsedPage],
        document_id: uuid.UUID,
        filename: str,
    ) -> list[ChunkData]:
        """
        Converts parsed pages into token-aware chunks.

        Args:
            pages_data: Output from the parser.
            document_id: Database ID of the document.
            filename: Original filename.

        Returns:
            List[ChunkData]
        """

        chunks: list[ChunkData] = []
        global_chunk_index = 0

        for page in pages_data:

            if not page.text.strip():
                continue

            text_chunks = self.splitter.split_text(page.text)

            for chunk_text in text_chunks:

                chunks.append(
                    ChunkData(
                        document_id=document_id,
                        chunk_index=global_chunk_index,
                        text=chunk_text,
                        page_number=page.page_number,
                        chunk_metadata={
                            "source_filename": filename,
                            "page_number": page.page_number,
                            "chunk_index": global_chunk_index,
                        },
                    )
                )

                global_chunk_index += 1

        return chunks