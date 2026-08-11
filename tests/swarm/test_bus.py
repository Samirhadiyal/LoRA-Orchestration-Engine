import asyncio

import pytest
from redis.exceptions import RedisError

from app.swarm.bus import SwarmMessageBus
from app.swarm.schemas import SwarmResult, SwarmTask


@pytest.mark.asyncio
async def test_swarm_schemas_validation() -> None:
    """Verify SwarmTask and SwarmResult instantiation and defaults."""
    task = SwarmTask(
        task_id="task_101",
        execution_id="exec_001",
        step_id="step_1",
        required_capability="rag_search",
        payload={"query": "test query"},
    )
    assert task.task_id == "task_101"
    assert task.timeout_seconds == 30.0
    assert task.attempt == 1

    result = SwarmResult(
        task_id="task_101",
        execution_id="exec_001",
        step_id="step_1",
        worker_id="worker_alpha",
        success=True,
        result={"chunks": ["doc1", "doc2"]},
        execution_time_seconds=0.12,
    )
    assert result.success is True
    assert result.error is None


@pytest.mark.asyncio
async def test_bus_publish_and_ack_stream_task() -> None:
    """Verify publishing a task to a Redis Stream, reading it, and acknowledging it."""
    bus = SwarmMessageBus()
    stream_key = "swarm:test:stream"
    group_name = "test_group"
    consumer_name = "worker_1"

    client = await bus.get_client()

    # 0. Clean up any leftover keys from previous failed test runs
    await client.delete(stream_key)

    task = SwarmTask(
        task_id="task_stream_001",
        execution_id="exec_001",
        step_id="step_1",
        required_capability="lora_inference",
        payload={"model": "adapter_v1"},
    )

    # 1. Ensure stream & consumer group exist
    try:
        await client.xgroup_create(
            stream_key, group_name, id="0", mkstream=True
        )
    except RedisError:
        pass  # Group already exists

    # 2. Publish task to stream
    msg_id = await bus.publish_task(stream_key, task)
    assert msg_id is not None
    assert isinstance(msg_id, str)

    # 3. Read message via consumer group (places message into PEL)
    read_entries = await client.xreadgroup(
        groupname=group_name,
        consumername=consumer_name,
        streams={stream_key: ">"},
        count=1,
    )
    assert len(read_entries) > 0

    # 4. Acknowledge task (XACK)
    acked = await bus.acknowledge_task(stream_key, group_name, msg_id)
    assert acked is True

    # 5. Cleanup test stream
    await client.delete(stream_key)
    await bus.close()


@pytest.mark.asyncio
async def test_bus_pubsub_result_listener() -> None:
    """Verify listening for task results via Redis Pub/Sub."""
    bus = SwarmMessageBus()

    result = SwarmResult(
        task_id="task_pubsub_999",
        execution_id="exec_002",
        step_id="step_2",
        worker_id="worker_beta",
        success=True,
        result={"output": "completed"},
        execution_time_seconds=0.35,
    )

    # Simulate worker responding after brief delay
    async def mock_worker_reply() -> None:
        await asyncio.sleep(0.1)
        await bus.publish_result(result)

    asyncio.create_task(mock_worker_reply())

    received_result = await bus.listen_for_result(
        task_id="task_pubsub_999", timeout=2.0
    )

    assert received_result is not None
    assert received_result.task_id == "task_pubsub_999"
    assert received_result.worker_id == "worker_beta"
    assert received_result.result == {"output": "completed"}

    await bus.close()   