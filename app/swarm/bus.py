# Path: app/swarm/bus.py
import logging
import time
from typing import Any

import redis.asyncio as redis
from redis.exceptions import ResponseError

from app.swarm.schemas import SwarmResult, SwarmTask

logger = logging.getLogger(__name__)


class RedisSwarmBus:
    """
    Phase 6 Track A: Redis-Backed Swarm Message Bus.

    - Uses Redis Streams (XADD/XREADGROUP) for durable at-least-once task delivery.
    - Uses Redis Pub/Sub for fast, ephemeral result routing back to the SwarmHandler.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_url = redis_url
        self.client = redis.from_url(redis_url, decode_responses=True)
        self.task_stream_prefix = "swarm:tasks:"
        self.result_channel_prefix = "swarm:results:"

    async def publish_task(self, worker_id: str, payload: dict[str, Any]) -> str:
        """Appends a validated SwarmTask to a Redis Stream."""
        task = SwarmTask(**payload)
        stream_name = f"{self.task_stream_prefix}{worker_id}"

        await self.client.xadd(stream_name, {"payload": task.model_dump_json()})
        logger.info("Published task %s to stream %s", task.task_id, stream_name)
        return task.task_id

    async def listen_for_result(self, task_id: str, timeout: float = 30.0) -> dict[str, Any] | None:
        """Subscribes to a Redis Pub/Sub channel to await the worker's result."""
        pubsub = self.client.pubsub()
        channel = f"{self.result_channel_prefix}{task_id}"
        await pubsub.subscribe(channel)

        try:
            start_time = time.time()
            while time.time() - start_time < timeout:
                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if message and message["type"] == "message":
                    result = SwarmResult.model_validate_json(message["data"])
                    return result.model_dump()
            
            logger.warning("Timeout waiting for result on channel %s", channel)
            return None
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.close()

    async def listen_for_task(self, worker_id: str, timeout: float = 5.0) -> dict[str, Any] | None:
        """Consumes a pending task from the worker's Redis Stream using a Consumer Group."""
        stream_name = f"{self.task_stream_prefix}{worker_id}"
        group_name = "swarm_workers_group"

        try:
            # Idempotently create consumer group
            try:
                await self.client.xgroup_create(stream_name, group_name, mkstream=True)
            except ResponseError as e:
                if "BUSYGROUP" not in str(e):
                    raise

            # Block and read 1 task from the stream
            messages = await self.client.xreadgroup(
                groupname=group_name,
                consumername=worker_id,
                streams={stream_name: ">"},
                count=1,
                block=int(timeout * 1000)
            )

            if messages:
                _, records = messages[0]
                message_id, data = records[0]
                task = SwarmTask.model_validate_json(data["payload"])

                # Acknowledge task completion immediately (at-most-once semantics for now)
                await self.client.xack(stream_name, group_name, message_id)
                return task.model_dump()

        except (redis.RedisError, ConnectionError, TimeoutError, ValueError, RuntimeError) as exc:
            logger.error("Error listening for tasks on %s: %s", stream_name, exc)
            
        return None

    async def publish_result(self, result_payload: dict[str, Any]) -> bool:
        """Broadcasts a validated SwarmResult via Redis Pub/Sub."""
        result = SwarmResult(**result_payload)
        channel = f"{self.result_channel_prefix}{result.task_id}"
        
        receivers = await self.client.publish(channel, result.model_dump_json())
        return receivers > 0
