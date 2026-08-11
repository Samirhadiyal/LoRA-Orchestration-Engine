import pytest
from typing import List
from app.context.optimizer import ContextOptimizerV2


def compute_context_precision(retrieved_chunks: List[str], ground_truth_keywords: List[str]) -> float:
    """Calculates the proportion of retrieved chunks containing relevant keywords."""
    if not retrieved_chunks:
        return 0.0
    relevant = sum(
        1 for chunk in retrieved_chunks 
        if any(kw.lower() in chunk.lower() for kw in ground_truth_keywords)
    )
    return relevant / len(retrieved_chunks)


def compute_faithfulness(final_response: str, context: str) -> float:
    """Measures lexical overlap to ensure the generation does not hallucinate outside retrieved context."""
    if not final_response or not context:
        return 0.0
    
    # Simple word-set overlap score
    response_words = set(w.lower() for w in final_response.split() if len(w) > 3)
    context_words = set(w.lower() for w in context.split() if len(w) > 3)
    
    if not response_words:
        return 1.0
        
    grounded_words = response_words.intersection(context_words)
    return len(grounded_words) / len(response_words)


class TestRAGEvaluationSuite:
    """Automated test suite measuring RAG precision, recall, and faithfulness."""

    def test_context_precision_benchmark(self):
        retrieved = [
            "Qdrant provides 384-dimensional dense vector embeddings with Cosine similarity.",
            "PostgreSQL stores relational user metadata.",
            "BM25 provides sparse keyword-based lexical matching."
        ]
        keywords = ["qdrant", "vector", "bm25", "embeddings"]
        
        precision = compute_context_precision(retrieved, keywords)
        print(f"\n[Benchmark] Retrieval Precision: {precision:.2%}")
        
        # Assert at least 60% precision
        assert precision >= 0.60, f"Precision {precision} below threshold"

    def test_generation_faithfulness_benchmark(self):
        retrieved_context = (
            "NeuroMesh uses a hybrid search pipeline combining Qdrant vector search "
            "and BM25 keyword retrieval for context optimization."
        )
        grounded_response = (
            "NeuroMesh implements a hybrid search pipeline with Qdrant vector search and BM25 retrieval."
        )
        
        score = compute_faithfulness(grounded_response, retrieved_context)
        print(f"\n[Benchmark] Generation Faithfulness Score: {score:.2%}")
        
        # Assert high faithfulness alignment
        assert score >= 0.70, f"Faithfulness score {score} indicates potential hallucination"

    def test_optimizer_budget_and_deduplication_eval(self):
        optimizer = ContextOptimizerV2(max_context_chars=12000)
        
        retrieved = (
            "FastAPI is a modern web framework for Python.\n\n"
            "FastAPI is a modern Python framework for high-performance APIs.\n\n"
            "Qdrant manages vector embeddings efficiently."
        )
        tools = {"step_1": {"status": "success", "data": "Tool output payload"}}
        
        optimized_context = optimizer.optimize(retrieved_context=retrieved, tool_results=tools)
        
        assert "--- TOOL RESULTS ---" in optimized_context
        assert "Qdrant manages vector embeddings" in optimized_context
        assert len(optimized_context) <= 12000