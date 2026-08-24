import pytest

from app.swarm.registry import WorkerRegistry
from app.swarm.workers.mcp_worker import MCPWorker


class MockMessageBus:
    def __init__(self):
        self.published_results: list[dict] = []

    async def listen_for_task(self, worker_id: str, timeout: float = 5.0) -> dict | None:
        return None

    async def publish_result(self, result_payload: dict) -> bool:
        self.published_results.append(result_payload)
        return True


class MockMCPClient:
    async def call_tool(self, tool_name: str, arguments: dict, **kwargs) -> dict:
        return {
            "status": "success",
            "tool": tool_name,
            "result_data": f"Fetched data for {arguments}",
        }


@pytest.mark.asyncio
async def test_mcp_worker_registration_and_capabilities():
    """Verifies worker registers online with expected MCP/external tool capabilities."""
    registry = WorkerRegistry()
    worker = MCPWorker(
        worker_id="mcp-worker-01",
        registry=registry,
        message_bus=MockMessageBus(),
    )

    await worker.start()
    try:
        workers = await registry.get_all_workers()
        assert len(workers) == 1
        assert workers[0].worker_id == "mcp-worker-01"
        assert {"mcp", "external_api", "tools"}.issubset(workers[0].capabilities)
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_mcp_worker_process_task_success():
    """Verifies process_task executes tool call and returns structured output."""
    registry = WorkerRegistry()
    worker = MCPWorker(
        worker_id="mcp-worker-01",
        registry=registry,
        message_bus=MockMessageBus(),
        mcp_client=MockMCPClient(),
    )

    task_payload = {
        "task_id": "task-300",
        "step_id": "step-mcp-1",
        "action": "mcp_tool",
        "metadata": {
            "tool_name": "search_docs",
            "arguments": {"query": "NeuroMesh architecture"},
        },
    }

    initial_load = worker.get_current_load()
    result = await worker.process_task(task_payload)

    assert result["success"] is True
    assert result["task_id"] == "task-300"
    assert result["result"]["tool_name"] == "search_docs"
    assert result["result"]["output"]["status"] == "success"
    assert worker.get_current_load() == initial_load


@pytest.mark.asyncio
async def test_mcp_worker_handles_tool_error():
    """Verifies exceptions are caught safely and return success=False payload."""
    class FailingMCPClient:
        async def call_tool(self, tool_name: str, arguments: dict, **kwargs):
            raise RuntimeError(f"External API timeout calling {tool_name}")

    registry = WorkerRegistry()
    worker = MCPWorker(
        worker_id="mcp-worker-01",
        registry=registry,
        message_bus=MockMessageBus(),
        mcp_client=FailingMCPClient(),
    )

    task_payload = {
        "task_id": "task-301",
        "step_id": "step-mcp-2",
        "action": "mcp_tool",
        "metadata": {"tool_name": "broken_tool", "arguments": {}},
    }

    result = await worker.process_task(task_payload)
    assert result["success"] is False
    assert "External API timeout" in (result["error"] or "")