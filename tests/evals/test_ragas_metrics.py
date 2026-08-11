# Path: tests/evals/test_ragas_metrics.py
import pytest
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)


@pytest.mark.eval
def test_neuromesh_ragas_benchmarks(rag_benchmark_dataset: Dataset):
    """
    Phase 5 Track B: Automated RAGAS evaluation for 'neuromesh_knowledge'.
    Evaluates context precision, recall, answer relevancy, and faithfulness.
    """
    # Execute RAGAS evaluation
    results = evaluate(
        dataset=rag_benchmark_dataset,
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
        raise_exceptions=False,
    )

    # Convert evaluation result to dictionary
    scores = results.to_pandas().mean(numeric_only=True).to_dict()

    # Define quality thresholds for Track B RAG pipeline
    assert scores.get("context_precision", 0.0) >= 0.80, (
        f"Context Precision below threshold: {scores.get('context_precision')}"
    )
    assert scores.get("faithfulness", 0.0) >= 0.85, (
        f"Faithfulness below threshold: {scores.get('faithfulness')}"
    )
    assert scores.get("context_recall", 0.0) >= 0.80, (
        f"Context Recall below threshold: {scores.get('context_recall')}"
    )
    assert scores.get("answer_relevancy", 0.0) >= 0.80, (
        f"Answer Relevancy below threshold: {scores.get('answer_relevancy')}"
    )


@pytest.mark.eval
def test_retrieval_fallback_faithfulness():
    """
    Verifies that when Qdrant returns a 404/fallback string,
    faithfulness scoring doesn't break or generate hallucinated claims.
    """
    fallback_dataset = Dataset.from_dict(
        {
            "question": ["What is an uninitialized database collection?"],
            "contexts": [["No external knowledge base context available."]],
            "answer": ["No external knowledge base context available."],
            "ground_truth": ["No external knowledge base context available."],
        }
    )

    results = evaluate(
        dataset=fallback_dataset,
        metrics=[faithfulness],
        raise_exceptions=False,
    )
    assert results["faithfulness"] is not None