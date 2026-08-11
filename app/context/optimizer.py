import json
import logging
from typing import Dict, Any, Optional, List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("neuromesh.optimizer")


class ContextOptimizerV2:
    """
    V2 Semantic Context Optimizer: Uses TF-IDF cosine similarity for deduplication
    and strictly adheres to a 12,000 character budget, prioritizing tool results.
    """

    def __init__(self, max_context_chars: int = 12000):
        self.max_context_chars = max_context_chars
        # CPU-friendly vectorizer for semantic deduplication (spares GPU VRAM)
        self.vectorizer = TfidfVectorizer(stop_words="english")

    def optimize(
        self, retrieved_context: Optional[str], tool_results: Dict[str, Any]
    ) -> str:
        # 1. Format tool results (Highest Priority)
        tool_str = self._format_tool_results(tool_results)

        if not retrieved_context:
            return tool_str[: self.max_context_chars]

        # 2. Calculate remaining budget
        remaining_budget = self.max_context_chars - len(tool_str)
        if remaining_budget <= 0:
            logger.warning(
                "Tool results exceeded context budget. Dropping retrieved context entirely."
            )
            return tool_str[: self.max_context_chars]

        # 3. Chunk the retrieved context by paragraphs
        chunks = [
            c.strip() for c in retrieved_context.split("\n\n") if c.strip()
        ]

        if not chunks:
            return tool_str

        # 4. Semantic Deduplication (Drop chunks with >85% similarity)
        unique_chunks = self._semantic_deduplicate(chunks, threshold=0.85)

        # 5. Fit remaining chunks into character budget
        optimized_context = ""
        for chunk in unique_chunks:
            if len(optimized_context) + len(chunk) + 2 <= remaining_budget:
                optimized_context += f"{chunk}\n\n"
            else:
                logger.info(
                    "Context budget reached. Truncating remaining chunks."
                )
                break

        # 6. Combine tool output + deduplicated context
        final_context = (
            f"{tool_str}\n\n--- RETRIEVED KNOWLEDGE ---\n{optimized_context}".strip()
        )
        return final_context[: self.max_context_chars]

    def _format_tool_results(self, tool_results: Dict[str, Any]) -> str:
        if not tool_results:
            return ""
        try:
            return "--- TOOL RESULTS ---\n" + json.dumps(
                tool_results, indent=2, default=str
            )
        except Exception as e:
            logger.error(f"Failed to format tool results: {e}")
            return str(tool_results)

    def _semantic_deduplicate(
        self, chunks: List[str], threshold: float
    ) -> List[str]:
        if len(chunks) <= 1:
            return chunks

        try:
            tfidf_matrix = self.vectorizer.fit_transform(chunks)
            sim_matrix = cosine_similarity(tfidf_matrix)

            keep_indices = []
            for i in range(len(chunks)):
                is_duplicate = False
                for j in keep_indices:
                    if sim_matrix[i][j] > threshold:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    keep_indices.append(i)

            return [chunks[i] for i in keep_indices]
        except Exception as e:
            logger.error(
                f"TF-IDF deduplication failed, keeping original chunks: {e}"
            )
            return chunks