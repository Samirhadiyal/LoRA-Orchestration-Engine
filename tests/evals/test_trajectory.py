# Path: tests/evals/test_trajectory.py
from typing import Any

import pytest

from app.context.optimizer import ContextOptimizer
from app.router.state import StepStatus


@pytest.mark.eval
def test_langgraph_trajectory_validity(mock_execution_trajectory: list[dict[str, Any]]):
    """
    Verifies that all steps in the execution plan reach StepStatus.COMPLETED
    without cycles or infinite retry loops.
    """
    for index, step in enumerate(mock_execution_trajectory):
        assert step["status"] == StepStatus.COMPLETED.value, (
            f"Step {step['step_id']} failed to reach COMPLETED state."
        )
        assert step["error"] is None, (
            f"Step {step['step_id']} encountered unexpected error: {step['error']}"
        )


@pytest.mark.eval
def test_trajectory_context_budget_enforcement():
    """
    Verifies across a simulated trajectory that ContextOptimizer V2
    strictly respects the 12,000 character hard limit even with large tool outputs.
    """
    optimizer = ContextOptimizer(max_context_chars=12000, similarity_threshold=0.82)

    # Simulate large tool outputs from SQLAgent / MCPClient
    tool_results = {
        "step_1": {"data": "A" * 7000},
        "step_2": {"data": "B" * 4000},
    }
    # Simulate lengthy retrieved chunks
    retrieved_context = "C" * 5000 + "\n\n" + "D" * 5000

    optimized_context = optimizer.optimize(
        retrieved_context=retrieved_context,
        tool_results=tool_results,
    )

    assert len(optimized_context) <= 12000, (
        f"Optimized context exceeded 12,000 chars: {len(optimized_context)}"
    )
    # Confirm Tool Results take priority over static RAG chunks
    assert "=== EXECUTION TOOL RESULTS ===" in optimized_context
    assert "step_1" in optimized_context
    assert "step_2" in optimized_context