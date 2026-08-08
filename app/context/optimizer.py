# app/context/optimizer.py
import logging
import hashlib
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ContextOptimizer:
    def __init__(self, max_context_chars: int = 12000):
        """
        Initializes the optimizer. 
        max_context_chars: Proxy for token limits (12,000 chars is roughly 3,000 tokens).
        """
        self.max_context_chars = max_context_chars

    def _hash_content(self, content: str) -> str:
        """Generates a quick MD5 hash to identify duplicate text."""
        return hashlib.md5(content.strip().encode('utf-8')).hexdigest()

    def optimize(self, retrieved_context: Optional[str], tool_results: Dict[str, Any]) -> str:
        """
        Merges, deduplicates, and truncates context to fit safely inside the LLM prompt.
        """
        logger.info("Starting deterministic context optimization...")
        
        optimized_parts = []
        seen_hashes = set()
        current_length = 0

        # 1. Process Tool Results (Highest Priority - MUST be seen by the LLM)
        if tool_results:
            optimized_parts.append("--- EXECUTED TOOL RESULTS ---")
            for step_id, result in tool_results.items():
                # Normalize tool output to string
                content = f"Tool Data [{step_id}]: {str(result)}"
                content_hash = self._hash_content(content)
                
                if content_hash not in seen_hashes:
                    seen_hashes.add(content_hash)
                    optimized_parts.append(content)
                    current_length += len(content)

        # 2. Process Retrieved Context (Lower Priority - Subject to truncation)
        if retrieved_context and "--- RETRIEVED CONTEXT ---" in retrieved_context:
            optimized_parts.append("\n--- KNOWLEDGE BASE CONTEXT ---")
            
            # Split by the delimiter we created in the RetrievalHandler (B3)
            chunks = retrieved_context.split("[Source ")
            
            for chunk in chunks:
                if not chunk.strip() or chunk.startswith("---"):
                    continue
                
                formatted_chunk = f"[Source {chunk.strip()}"
                chunk_hash = self._hash_content(formatted_chunk)
                
                # Deduplicate chunks that might have been returned multiple times by hybrid search
                if chunk_hash not in seen_hashes:
                    # Enforce Context Budget
                    if current_length + len(formatted_chunk) > self.max_context_chars:
                        logger.warning("Context budget exceeded. Truncating remaining retrieved chunks.")
                        optimized_parts.append("... [Additional context truncated to preserve token budget]")
                        break
                    
                    seen_hashes.add(chunk_hash)
                    optimized_parts.append(formatted_chunk)
                    current_length += len(formatted_chunk)
        
        # Fallback if no context was found
        if not optimized_parts:
            return "No additional external context or tool data available."

        logger.info(f"Context optimization complete. Final size: {current_length} characters.")
        return "\n".join(optimized_parts)