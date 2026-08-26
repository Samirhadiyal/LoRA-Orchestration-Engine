# app/reasoning/citations.py
"""
Phase 5: Citation Generator — Maps claims in the generated answer back to
source document chunks using TF-IDF similarity scoring.
"""

import logging
import re
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class CitationGenerator:
    """
    Maps sentences in a generated answer back to the source context chunks
    that support them, producing confidence-scored citations.
    """

    def __init__(self, min_confidence: float = 0.15) -> None:
        """
        Args:
            min_confidence: Minimum cosine similarity to consider a chunk as a
                            supporting source. Lower = more permissive.
        """
        self.min_confidence = min_confidence
        self._vectorizer = TfidfVectorizer(
            stop_words="english", lowercase=True, ngram_range=(1, 2)
        )

    def generate(
        self,
        answer: str,
        retrieved_chunks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Produces a list of citation objects linking answer sentences to source chunks.

        Args:
            answer: The final generated answer text.
            retrieved_chunks: List of dicts with at least a "text" key, and optionally
                              "chunk_id", "doc_id", and "metadata".

        Returns:
            List of citation dicts:
                [{"claim": "...", "source_chunk_id": "...", "doc_id": "...",
                  "source_text": "...", "confidence": 0.85}, ...]
        """
        if not answer or not retrieved_chunks:
            return []

        # 1. Split the answer into individual sentences
        sentences = self._split_sentences(answer)
        if not sentences:
            return []

        # 2. Extract chunk texts and IDs
        chunk_texts = [chunk.get("text", "") for chunk in retrieved_chunks]
        chunk_texts = [t for t in chunk_texts if t.strip()]

        if not chunk_texts:
            return []

        # 3. Build TF-IDF matrix over chunks + sentences
        try:
            all_texts = chunk_texts + sentences
            tfidf_matrix = self._vectorizer.fit_transform(all_texts)

            chunk_vectors = tfidf_matrix[: len(chunk_texts)]
            sentence_vectors = tfidf_matrix[len(chunk_texts) :]

            sim_matrix = cosine_similarity(sentence_vectors, chunk_vectors)
        except ValueError:
            logger.warning("TF-IDF vectorization failed; returning empty citations.")
            return []

        # 4. For each sentence, find the best-matching chunk
        citations = []
        for sent_idx, sentence in enumerate(sentences):
            best_chunk_idx = int(sim_matrix[sent_idx].argmax())
            best_score = float(sim_matrix[sent_idx][best_chunk_idx])

            if best_score < self.min_confidence:
                continue

            source_chunk = retrieved_chunks[best_chunk_idx] if best_chunk_idx < len(retrieved_chunks) else {}

            citations.append(
                {
                    "claim": sentence.strip(),
                    "source_chunk_id": source_chunk.get("chunk_id", f"chunk_{best_chunk_idx}"),
                    "doc_id": source_chunk.get("doc_id", ""),
                    "source_text": chunk_texts[best_chunk_idx][:200],  # Truncate for readability
                    "confidence": round(best_score, 4),
                }
            )

        # Sort by confidence descending
        citations.sort(key=lambda c: c["confidence"], reverse=True)
        return citations

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """
        Splits text into sentences using regex. Handles common abbreviations.
        """
        # Split on sentence-ending punctuation followed by whitespace
        raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        # Filter out very short fragments (likely not real sentences)
        return [s for s in raw_sentences if len(s.split()) >= 4]
