# Path: app/context/optimizer.py
import json
import logging
from typing import Any, Dict, List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class ContextOptimizer:
    """
    Phase 5 Track B: Semantic Context Optimizer V2.
    - CPU-bound TF-IDF + Cosine Similarity semantic deduplication.
    - Enforces 12,000 max character budget.
    - Strictly prioritizes Tool Results over static RAG chunks.
    """

    def __init__(
        self, max_context_chars: int = 12000, similarity_threshold: float = 0.82
    ):
        self.max_context_chars = max_context_chars
        self.similarity_threshold = similarity_threshold
        self._vectorizer = TfidfVectorizer(
            stop_words="english", lowercase=True, ngram_range=(1, 2)
        )

    def optimize(
        self,
        retrieved_context: Optional[str],
        tool_results: Dict[str, Any],
    ) -> str:
        """
        Deterministic V2 implementation: formats tool results first, deduplicates
        retrieved context chunks semantically on CPU, and packs within budget.
        """
        # 1. Serialize and prioritize Tool Results (Highest Priority)
        tool_section = self._format_tool_results(tool_results)
        remaining_budget = self.max_context_chars - len(tool_section)

        if remaining_budget <= 0:
            logger.warning(
                "Tool results exceeded max_context_chars budget. Truncating tool output."
            )
            return tool_section[: self.max_context_chars]

        if not retrieved_context or not retrieved_context.strip():
            return tool_section.strip()

        # 2. Split retrieved context into distinct chunks (paragraphs / blocks)
        raw_chunks = [
            c.strip()
            for c in retrieved_context.split("\n\n")
            if len(c.strip()) > 20
        ]
        if not raw_chunks:
            return f"{tool_section}\n\n{retrieved_context}"[: self.max_context_chars].strip()

        # 3. CPU-Bound Semantic Deduplication
        unique_chunks = self._semantic_deduplicate(raw_chunks)

        # 4. Pack unique chunks into remaining budget
        packed_rag = self._pack_chunks_to_budget(unique_chunks, remaining_budget)

        final_context = f"{tool_section}\n\n=== RETRIEVED KNOWLEDGE ===\n{packed_rag}"
        return final_context.strip()

    def _format_tool_results(self, tool_results: Dict[str, Any]) -> str:
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
            except Exception:
                serialized_out = str(output)
            formatted_lines.append(f"[{step_id}]: {serialized_out}")
        return "\n".join(formatted_lines)

    def _semantic_deduplicate(self, chunks: List[str]) -> List[str]:
        if len(chunks) <= 1:
            return chunks

        try:
            tfidf_matrix = self._vectorizer.fit_transform(chunks)
            sim_matrix = cosine_similarity(tfidf_matrix)
        except ValueError:
            # Fallback if text is too short or contains only stop words
            return list(dict.fromkeys(chunks))

        kept_indices = []
        for i in range(len(chunks)):
            is_duplicate = False
            for kept_idx in kept_indices:
                if sim_matrix[i][kept_idx] >= self.similarity_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept_indices.append(i)

        return [chunks[idx] for idx in kept_indices]

    def _pack_chunks_to_budget(self, chunks: List[str], budget: int) -> str:
        packed_lines = []
        current_len = 0

        for chunk in chunks:
            chunk_len = len(chunk) + 2  # account for \n\n separator
            if current_len + chunk_len <= budget:
                packed_lines.append(chunk)
                current_len += chunk_len
            else:
                # Budget exhausted; stop packing
                break

        return "\n\n".join(packed_lines)