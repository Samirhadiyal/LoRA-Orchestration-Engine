# Path: tests/swarm/test_sql_worker.py
import pytest

from app.swarm.registry import WorkerRegistry
from app.swarm.workers.sql_worker import SQLWorker


class MockMessageBus:
    def __init__(self):
        self.published_results: list[dict] = []

    async def listen_for_task(self, worker_id: str, timeout: float = 5.0) -> dict | None:
        return None

    async def publish_result(self, result_payload: dict) -> bool:
        self.published_results.append(result_payload)
        return True


class MockSQLEngine:
    async def execute_query(self, query: str, **kwargs) -> list[dict]:
        return [
            {"id": 1, "department": "AI Research", "budget": 50000},
            {"id": 2, "department": "Systems Engineering", "budget": 65000},
        ]


@pytest.mark.asyncio
async def test_sql_worker_registration_and_capabilities():
    """Verifies worker registers online with expected SQL/database capabilities."""
    registry = WorkerRegistry()
    worker = SQLWorker(
        worker_id="sql-worker-01",
        registry=registry,
        message_bus=MockMessageBus(),
    )

    await worker.start()
    try:
        workers = await registry.get_all_workers()
        assert len(workers) == 1
        assert workers[0].worker_id == "sql-worker-01"
        assert {"sql", "database", "analytics"}.issubset(workers[0].capabilities)
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_sql_worker_process_task_success():
    """Verifies process_task executes database query and returns structured rows."""
    registry = WorkerRegistry()
    worker = SQLWorker(
        worker_id="sql-worker-01",
        registry=registry,
        message_bus=MockMessageBus(),
        sql_engine=MockSQLEngine(),
    )

    task_payload = {
        "task_id": "task-200",
        "step_id": "step-sql-1",
        "action": "sql_query",
        "metadata": {"query": "SELECT * FROM budgets", "analytics": True},
    }

    initial_load = worker.get_current_load()
    result = await worker.process_task(task_payload)

    assert result["success"] is True
    assert result["task_id"] == "task-200"
    assert result["result"]["row_count"] == 2
    assert len(result["result"]["rows"]) == 2
    assert worker.get_current_load() == initial_load


@pytest.mark.asyncio
async def test_sql_worker_handles_query_error():
    """Verifies exceptions are caught safely and return success=False payload."""
    class FailingSQLEngine:
        async def execute_query(self, query: str, **kwargs):
            raise ValueError("SQL syntax error near SELECT")

    registry = WorkerRegistry()
    worker = SQLWorker(
        worker_id="sql-worker-01",
        registry=registry,
        message_bus=MockMessageBus(),
        sql_engine=FailingSQLEngine(),
    )

    task_payload = {
        "task_id": "task-201",
        "step_id": "step-sql-2",
        "action": "sql_query",
        "metadata": {"query": "INVALID SQL"},
    }

    result = await worker.process_task(task_payload)
    assert result["success"] is False
    assert "SQL syntax error" in (result["error"] or "")