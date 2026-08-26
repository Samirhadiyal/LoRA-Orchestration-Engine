# Path: tests/swarm/test_redis_bus.py
import pytest
from unittest.mock import AsyncMock, patch
from app.swarm.bus import RedisSwarmBus


@pytest.fixture
def mock_redis():
    with patch("app.swarm.bus.redis.from_url") as mock_from_url:
        mock_client = AsyncMock()
        mock_from_url.return_value = mock_client
        yield mock_client


@pytest.mark.asyncio
async def test_redis_bus_publish_task(mock_redis):
    """Verifies tasks are validated and published to the correct Redis stream."""
    bus = RedisSwarmBus(redis_url="redis://fake:6379/0")
    
    payload = {
        "task_id": "test-task-1",
        "step_id": "step-1",
        "action": "rag_search",
        "metadata": {"query": "AI docs"}
    }
    
    task_id = await bus.publish_task(worker_id="rag-worker", payload=payload)
    
    assert task_id == "test-task-1"
    mock_redis.xadd.assert_called_once()
    
    # Verify the stream name includes the worker_id
    args, _ = mock_redis.xadd.call_args
    assert args[0] == "swarm:tasks:rag-worker"


@pytest.mark.asyncio
async def test_redis_bus_publish_result(mock_redis):
    """Verifies results are validated and published to the correct Pub/Sub channel."""
    bus = RedisSwarmBus(redis_url="redis://fake:6379/0")
    
    result_payload = {
        "task_id": "test-task-2",
        "step_id": "step-2",
        "worker_id": "rag-worker",
        "success": True,
        "result": {"data": "found"}
    }
    
    # Mock that 1 receiver is listening on the pubsub channel
    mock_redis.publish.return_value = 1
    
    success = await bus.publish_result(result_payload)
    
    assert success is True
    mock_redis.publish.assert_called_once()
    
    # Verify the channel name matches the task_id
    args, _ = mock_redis.publish.call_args
    assert args[0] == "swarm:results:test-task-2"
