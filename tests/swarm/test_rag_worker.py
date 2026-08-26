# Path: tests/swarm/test_rag_worker.py
import pytest

from app.swarm.registry import WorkerRegistry
from app.swarm.workers.rag_worker import RAGWorker


class MockMessageBus:
    def __init__(self):
        self.published_results: list[dict] = []

    async def listen_for_task(self, worker_id: str, timeout: float = 5.0) -> dict | None:
        return None

    async def publish_result(self, result_payload: dict) -> bool:
        self.published_results.append(result_payload)
        return True


class MockRetrievalEngine:
    async def search(self, query: str, top_k: int, **kwargs) -> list[dict]:
        return [
            {
                "id": f"mock-{i}",
                "content": f"Result {i} for {query}",
                "score": 0.99,
                "source": "mock_db",
            }
            for i in range(top_k)
        ]


@pytest.mark.asyncio
async def test_rag_worker_registration_and_capabilities():
    """Verifies worker registers online with expected RAG capabilities."""
    registry = WorkerRegistry()
    worker = RAGWorker(
        worker_id="rag-worker-01",
        registry=registry,
        message_bus=MockMessageBus(),
    )

    await worker.start()
    try:
        workers = await registry.get_all_workers()
        assert len(workers) == 1
        assert workers[0].worker_id == "rag-worker-01"
        assert {"rag", "retrieval", "embed"}.issubset(workers[0].capabilities)
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_rag_worker_process_task_success():
    """Verifies process_task executes search and returns structured chunks."""
    registry = WorkerRegistry()
    worker = RAGWorker(
        worker_id="rag-worker-01",
        registry=registry,
        message_bus=MockMessageBus(),
        retrieval_engine=MockRetrievalEngine(),
    )

    task_payload = {
        "task_id": "task-100",
        "step_id": "step-1",
        "action": "rag_search",
        "metadata": {"query": "How does LoRA work?", "top_k": 2},
    }

    initial_load = worker.get_current_load()
    result = await worker.process_task(task_payload)

    assert result["success"] is True
    assert result["task_id"] == "task-100"
    assert result["result"]["retrieved_count"] == 2
    assert len(result["result"]["chunks"]) == 2
    # Verify load cools down after completion
    assert worker.get_current_load() == initial_load


@pytest.mark.asyncio
async def test_rag_worker_handles_retrieval_error():
    """Verifies exceptions are caught safely and return success=False payload."""
    class FailingEngine:
        async def search(self, query: str, top_k: int, **kwargs):
            raise RuntimeError("Qdrant connection dropped")

    registry = WorkerRegistry()
    worker = RAGWorker(
        worker_id="rag-worker-01",
        registry=registry,
        message_bus=MockMessageBus(),
        retrieval_engine=FailingEngine(),
    )

    task_payload = {
        "task_id": "task-101",
        "step_id": "step-2",
        "action": "rag_search",
        "metadata": {"query": "test", "top_k": 5},
    }

    result = await worker.process_task(task_payload)
    assert result["success"] is False
    assert "Qdrant connection dropped" in (result["error"] or "")