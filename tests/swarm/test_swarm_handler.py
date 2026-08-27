# Path: tests/swarm/test_swarm_handler.py
import uuid

import pytest

from app.planner.schemas import ExecutionPlan
from app.router.handlers import SwarmHandler
from app.router.state import ExecutionState, ExecutionStep, StepStatus


class MockMessageBus:
    def __init__(self, response_payload: dict | None = None):
        self.response_payload = response_payload or {"success": True, "result": {"output": "swarm_ok"}}
        self.published_tasks: list[dict] = []

    async def publish_task(self, worker_id: str, payload: dict) -> str:
        self.published_tasks.append(payload)
        return payload["task_id"]

    async def listen_for_result(self, task_id: str, timeout: float = 30.0) -> dict | None:
        return self.response_payload


@pytest.mark.asyncio
async def test_swarm_handler_disabled_by_default(monkeypatch):
    """Verifies SwarmHandler fails safely when NEUROMESH_SWARM_ENABLED is false/unset."""
    monkeypatch.setenv("NEUROMESH_SWARM_ENABLED", "false")
    handler = SwarmHandler(message_bus=MockMessageBus())
    plan = ExecutionPlan(user_intent="test", requires_retrieval=False, steps=[])
    state = ExecutionState(request_id=uuid.uuid4(), user_query="test query", plan=plan)
    step = ExecutionStep(step_id="s1", action="swarm_delegate", status=StepStatus.PENDING)

    updated_state = await handler.execute(state, step)
    assert step.status == StepStatus.FAILED
    assert "disabled" in (updated_state.error or "").lower()


@pytest.mark.asyncio
async def test_swarm_handler_executes_when_enabled(monkeypatch):
    """Verifies SwarmHandler dispatches task and updates state when enabled."""
    monkeypatch.setenv("NEUROMESH_SWARM_ENABLED", "true")
    mock_bus = MockMessageBus({"success": True, "result": {"data": "processed by worker"}})
    handler = SwarmHandler(message_bus=mock_bus)

    plan = ExecutionPlan(user_intent="test", requires_retrieval=False, steps=[])
    state = ExecutionState(request_id=uuid.uuid4(), user_query="test query", plan=plan, tool_results=[])
    step = ExecutionStep(
        step_id="s1",
        action="swarm_delegate",
        status=StepStatus.PENDING,
        metadata={"heavy_rag": True},
    )

    updated_state = await handler.execute(state, step)
    assert step.status == StepStatus.COMPLETED
    assert len(mock_bus.published_tasks) == 1
    assert updated_state.tool_results[0].get("s1") == {"data": "processed by worker"}