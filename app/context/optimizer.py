import json
import logging
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class ContextOptimizer:
    """
    Phase 5 Track B: Semantic Context Optimizer V2.

    Uses CPU-bound TF-IDF + cosine similarity deduplication so GPU VRAM stays
    reserved for 4-bit QLoRA generation.
    """

    def __init__(
        self, max_context_chars: int = 12000, similarity_threshold: float = 0.82
    ) -> None:
        self.max_context_chars = max_context_chars
        self.similarity_threshold = similarity_threshold
        self._vectorizer = TfidfVectorizer(
            stop_words="english", lowercase=True, ngram_range=(1, 2)
        )

    def optimize(
        self,
        retrieved_context: str | None,
        tool_results: dict[str, Any],
    ) -> str:
        """
        Format tool results first, semantically deduplicate retrieved chunks,
        and pack the final context within the hard character budget.
        """
        tool_section = self._format_tool_results(tool_results)
        remaining_budget = self.max_context_chars - len(tool_section)

        if remaining_budget <= 0:
            logger.warning(
                "Tool results exceeded max_context_chars budget. Truncating tool output."
            )
            return tool_section[: self.max_context_chars]

        if not retrieved_context or not retrieved_context.strip():
            return tool_section.strip()

        raw_chunks = [
            chunk.strip()
            for chunk in retrieved_context.split("\n\n")
            if len(chunk.strip()) > 20
        ]
        if not raw_chunks:
            return f"{tool_section}\n\n{retrieved_context}"[
                : self.max_context_chars
            ].strip()

        unique_chunks = self._semantic_deduplicate(raw_chunks)
        packed_rag = self._pack_chunks_to_budget(unique_chunks, remaining_budget)

        final_context = f"{tool_section}\n\n=== RETRIEVED KNOWLEDGE ===\n{packed_rag}"
        return final_context[: self.max_context_chars].strip()

    def _format_tool_results(self, tool_results: dict[str, Any]) -> str:
        if not tool_results:
            return ""

        formatted_lines = ["=== EXECUTION TOOL RESULTS ==="]
        for step_id, output in tool_results.items():
            try:
                serialized_out = (
                    json.dumps(output, ensure_ascii=False)
                    if isinstance(output, (dict, list))
                    else str(output)
                )
            except TypeError:
                serialized_out = str(output)
            formatted_lines.append(f"[{step_id}]: {serialized_out}")

        return "\n".join(formatted_lines)

    def _semantic_deduplicate(self, chunks: list[str]) -> list[str]:
        if len(chunks) <= 1:
            return chunks

        try:
            tfidf_matrix = self._vectorizer.fit_transform(chunks)
            sim_matrix = cosine_similarity(tfidf_matrix)
        except ValueError:
            return list(dict.fromkeys(chunks))

        kept_indices: list[int] = []
        for index in range(len(chunks)):
            is_duplicate = any(
                sim_matrix[index][kept_index] >= self.similarity_threshold
                for kept_index in kept_indices
            )
            if not is_duplicate:
                kept_indices.append(index)

        return [chunks[index] for index in kept_indices]

    def _pack_chunks_to_budget(self, chunks: list[str], budget: int) -> str:
        packed_lines = []
        current_len = 0

        for chunk in chunks:
            chunk_len = len(chunk) + 2
            if current_len + chunk_len > budget:
                break

            packed_lines.append(chunk)
            current_len += chunk_len

        return "\n\n".join(packed_lines)
