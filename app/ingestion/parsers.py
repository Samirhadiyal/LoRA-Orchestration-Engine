# app/ingestion/parsers.py

import io
from typing import List

import docx
import pypdf

from app.ingestion.schemas import ParsedPage


class DocumentParser:
    @staticmethod
    def parse_pdf(file_bytes: bytes) -> List[ParsedPage]:
        """
        Extract text from a PDF while preserving page numbers.
        """

        try:
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        except Exception as e:
            raise ValueError(f"Failed to parse PDF: {e}") from e

        pages_data: List[ParsedPage] = []

        for i, page in enumerate(reader.pages):
            text = page.extract_text()

            if text and text.strip():
                pages_data.append(
                    ParsedPage(
                        text=text.strip(),
                        page_number=i + 1,
                    )
                )

        return pages_data

    @staticmethod
    def parse_docx(file_bytes: bytes) -> List[ParsedPage]:
        """
        Extract text from a DOCX document.
        DOCX does not expose reliable page numbers,
        so the entire document is treated as page 1.
        """

        try:
            doc = docx.Document(io.BytesIO(file_bytes))
        except Exception as e:
            raise ValueError(f"Failed to parse DOCX: {e}") from e

        full_text = "\n".join(
            para.text
            for para in doc.paragraphs
            if para.text.strip()
        )

        return [
            ParsedPage(
                text=full_text.strip(),
                page_number=1,
            )
        ]

    @staticmethod
    def parse_txt(file_bytes: bytes) -> List[ParsedPage]:
        """
        Extract text from TXT, Markdown, CSV and other plain text files.
        """

        try:
            text = file_bytes.decode("utf-8", errors="replace")
        except Exception as e:
            raise ValueError(f"Failed to parse text file: {e}") from e

        return [
            ParsedPage(
                text=text.strip(),
                page_number=1,
            )
        ]

    @classmethod
    def parse(cls, file_bytes: bytes, mime_type: str) -> List[ParsedPage]:
        """
        Route the document to the correct parser based on MIME type.
        """

        if mime_type == "application/pdf":
            return cls.parse_pdf(file_bytes)

        elif mime_type in {
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword",
        }:
            return cls.parse_docx(file_bytes)

        elif mime_type.startswith("text/"):
            return cls.parse_txt(file_bytes)

        raise ValueError(f"Unsupported MIME type: {mime_type}")