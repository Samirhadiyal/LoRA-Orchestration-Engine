import asyncio
import logging
import time
from typing import Any

import pytest

from app.router.handlers import SwarmHandler
from app.router.state import ExecutionState, ExecutionStep, StepStatus
from app.swarm.registry import WorkerRegistry
from app.swarm.workers.mcp_worker import MCPWorker
from app.swarm.workers.rag_worker import RAGWorker
from app.swarm.workers.sql_worker import SQLWorker

logger = logging.getLogger(__name__)


class InMemorySwarmBus:
    """
    Phase 6D E2E Test Bus:
    Bridges SwarmHandler publish/listen calls directly to WorkerRegistry-selected
    workers via background asyncio tasks, simulating real distributed queue dispatch.
    """

    def __init__(self, registry: WorkerRegistry, workers_map: dict[str, Any]):
        self.registry = registry
        self.workers_map = workers_map
        self.results: dict[str, dict[str, Any]] = {}
        self._action_capability_map = {
            "rag_search": ["rag"],
            "sql_query": ["sql"],
            "mcp_tool": ["mcp"],
            "external_api": ["mcp"],
        }

    async def publish_task(self, worker_id: str, payload: dict[str, Any]) -> str:
        task_id = str(payload.get("task_id", "unknown"))
        action = str(payload.get("action", "")).lower()
        required_caps = self._action_capability_map.get(action, ["rag"])

        # Select healthy worker with lowest load for the required capability
        selected_profile = await self.registry.select_worker(required_caps)
        if not selected_profile:
            self.results[task_id] = {
                "success": False,
                "error": f"No eligible worker found for capabilities {required_caps}",
            }
            return task_id

        worker = self.workers_map.get(selected_profile.worker_id)
        if not worker:
            self.results[task_id] = {
                "success": False,
                "error": f"Worker instance '{selected_profile.worker_id}' not found",
            }
            return task_id

        # Execute task concurrently in background
        asyncio.create_task(self._run_worker_task(worker, task_id, payload))
        return task_id

    async def _run_worker_task(self, worker: Any, task_id: str, payload: dict[str, Any]) -> None:
        try:
            result = await worker.process_task(payload)
            self.results[task_id] = result
        except (RuntimeError, ValueError, OSError, TypeError) as exc:
            self.results[task_id] = {"success": False, "error": str(exc)}

    async def listen_for_result(self, task_id: str, timeout: float = 5.0) -> dict[str, Any] | None:
        start = time.time()
        while time.time() - start < timeout:
            if task_id in self.results:
                return self.results.pop(task_id)
            await asyncio.sleep(0.01)
        return {"success": False, "error": f"Task '{task_id}' timed out after {timeout}s"}


class LatencyMockEngine:
    """Simulates real-world I/O latency (0.10s) to measure parallel speedups accurately."""

    def __init__(self, delay_seconds: float = 0.10):
        self.delay_seconds = delay_seconds

    async def search(self, query: str, top_k: int, **kwargs) -> list[dict[str, Any]]:
        await asyncio.sleep(self.delay_seconds)
        return [{"id": "rag-1", "content": f"Context for '{query}'"}]

    async def execute_query(self, query: str, **kwargs) -> list[dict[str, Any]]:
        await asyncio.sleep(self.delay_seconds)
        return [{"metric": "revenue", "value": 100000}]

    async def call_tool(self, tool_name: str, arguments: dict[str, Any], **kwargs) -> dict[str, Any]:
        await asyncio.sleep(self.delay_seconds)
        return {"tool": tool_name, "status": "ok"}


@pytest.fixture
async def swarm_environment():
    """Sets up registry, mock bus, and 3 specialized workers with simulated 100ms I/O latency."""
    registry = WorkerRegistry()
    engine = LatencyMockEngine(delay_seconds=0.10)

    workers_map: dict[str, Any] = {}
    bus = InMemorySwarmBus(registry=registry, workers_map=workers_map)

    rag = RAGWorker("rag-01", registry=registry, message_bus=bus, retrieval_engine=engine)
    sql = SQLWorker("sql-01", registry=registry, message_bus=bus, sql_engine=engine)
    mcp = MCPWorker("mcp-01", registry=registry, message_bus=bus, mcp_client=engine)

    workers_map["rag-01"] = rag
    workers_map["sql-01"] = sql
    workers_map["mcp-01"] = mcp

    await rag.start()
    await sql.start()
    await mcp.start()

    yield registry, bus, handler_from_bus(bus)

    await rag.stop()
    await sql.stop()
    await mcp.stop()


def handler_from_bus(bus: InMemorySwarmBus) -> SwarmHandler:
    return SwarmHandler(message_bus=bus, default_timeout_seconds=5.0)


@pytest.mark.asyncio
async def test_e2e_swarm_router_delegation(swarm_environment, monkeypatch):
    """
    Verifies SwarmHandler routes different action types to the correct domain worker
    and applies canonical state updates.
    """
    monkeypatch.setenv("NEUROMESH_SWARM_ENABLED", "true")
    _, _, handler = swarm_environment

    import uuid

    from app.planner.schemas import ExecutionPlan
    dummy_plan = ExecutionPlan(user_intent="test", requires_retrieval=False, steps=[])
    state = ExecutionState(request_id=uuid.uuid4(), plan=dummy_plan, user_query="Run full analysis", tool_results=[])
    steps = [
        ExecutionStep(step_id="step-rag", action="rag_search", status=StepStatus.PENDING, metadata={"query": "AI docs"}),
        ExecutionStep(step_id="step-sql", action="sql_query", status=StepStatus.PENDING, metadata={"query": "SELECT * FROM sales"}),
        ExecutionStep(step_id="step-mcp", action="mcp_tool", status=StepStatus.PENDING, metadata={"tool_name": "weather_api", "arguments": {}}),
    ]

    for step in steps:
        state = await handler.execute(state, step)
        assert step.status == StepStatus.COMPLETED
        assert any(step.step_id in res for res in state.tool_results)
        assert len(state.tool_results) > 0


@pytest.mark.asyncio
async def test_e2e_parallel_swarm_benchmark(swarm_environment, monkeypatch):
    """
    Phase 6D Stress Benchmark:
    Dispatches 6 tasks (2 RAG, 2 SQL, 2 MCP), each taking 100ms sequentially.
    Proves parallel swarm execution completes in <0.35s (expected sequential: ~0.60s),
    achieving a > 1.8x concurrent speedup.
    """
    monkeypatch.setenv("NEUROMESH_SWARM_ENABLED", "true")
    _, _, handler = swarm_environment

    steps = [
        ExecutionStep(step_id="r1", action="rag_search", status=StepStatus.PENDING, metadata={"query": "q1"}),
        ExecutionStep(step_id="r2", action="rag_search", status=StepStatus.PENDING, metadata={"query": "q2"}),
        ExecutionStep(step_id="s1", action="sql_query", status=StepStatus.PENDING, metadata={"query": "SELECT 1"}),
        ExecutionStep(step_id="s2", action="sql_query", status=StepStatus.PENDING, metadata={"query": "SELECT 2"}),
        ExecutionStep(step_id="m1", action="mcp_tool", status=StepStatus.PENDING, metadata={"tool_name": "t1", "arguments": {}}),
        ExecutionStep(step_id="m2", action="mcp_tool", status=StepStatus.PENDING, metadata={"tool_name": "t2", "arguments": {}}),
    ]

    import uuid

    from app.planner.schemas import ExecutionPlan
    dummy_plan = ExecutionPlan(user_intent="test", requires_retrieval=False, steps=[])
    
    # Execute all 6 tasks concurrently
    start_time = time.time()
    await asyncio.gather(*(
        handler.execute(ExecutionState(request_id=uuid.uuid4(), plan=dummy_plan, user_query="parallel test", tool_results=[]), step)
        for step in steps
    ))
    parallel_duration = time.time() - start_time

    # Sequential baseline: 6 tasks * 0.10s I/O delay = ~0.60s
    expected_sequential = 6 * 0.10
    speedup = expected_sequential / max(0.001, parallel_duration)

    logger.info("Swarm E2E Benchmark: Parallel=%.3fs | Expected Sequential=%.3fs | Speedup=%.2fx", parallel_duration, expected_sequential, speedup)

    for step in steps:
        assert step.status == StepStatus.COMPLETED

    # Assert parallel execution beat sequential execution significantly
    assert parallel_duration < 0.35, f"Parallel execution took too long: {parallel_duration:.3f}s"
    assert speedup >= 1.8, f"Expected at least 1.8x speedup, got {speedup:.2f}x"