<<<<<<< HEAD
import logging
import time

import redis.asyncio as aioredis
from pydantic import ValidationError
from redis.exceptions import RedisError
=======
﻿# Path: app/swarm/bus.py
import logging
import time
from typing import Any

import redis.asyncio as redis
from redis.exceptions import ResponseError
>>>>>>> c6f8ca0621e09e8e2ff4bedd0ad32bc0ddbc07f2

from app.swarm.schemas import SwarmResult, SwarmTask

logger = logging.getLogger(__name__)


<<<<<<< HEAD
class SwarmMessageBus:
    """Async Redis Streams & Pub/Sub transport layer for Swarm orchestration."""

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_url = redis_url
        self._redis: aioredis.Redis | None = None

    async def get_client(self) -> aioredis.Redis:
        """Lazy-load and return the async Redis client."""
        if self._redis is None:
            self._redis = aioredis.from_url(
                self.redis_url, decode_responses=True
            )
        return self._redis

    async def close(self) -> None:
        """Close the Redis connection pool cleanly."""
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None

    async def publish_task(self, stream_key: str, task: SwarmTask) -> str:
        """Publish a SwarmTask to a specified Redis Stream via XADD."""
        client = await self.get_client()
        task_data = {"payload": task.model_dump_json()}

        message_id = await client.xadd(stream_key, task_data)  # type: ignore[arg-type]
        logger.info(
            "Published SwarmTask %s to stream %s (Msg ID: %s)",
            task.task_id,
            stream_key,
            message_id,
        )
        return str(message_id)

    async def publish_result(self, result: SwarmResult) -> None:
        """Publish a SwarmResult over Pub/Sub channel for listening clients."""
        client = await self.get_client()
        channel = f"swarm:results:{result.task_id}"
        await client.publish(channel, result.model_dump_json())
        logger.debug("Published SwarmResult to channel %s", channel)

    async def listen_for_result(
        self, task_id: str, timeout: float = 30.0
    ) -> SwarmResult | None:
        """Listen for a SwarmResult on a dedicated Pub/Sub channel with timeout."""
        client = await self.get_client()
        pubsub = client.pubsub()
        channel = f"swarm:results:{task_id}"

        await pubsub.subscribe(channel)
        start_time = time.time()

        try:
            while (time.time() - start_time) < timeout:
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=1.0
                )
                if message and message.get("type") == "message":
                    data = message.get("data")
                    if isinstance(data, str):
                        try:
                            return SwarmResult.model_validate_json(data)
                        except ValidationError as err:
                            logger.error(
                                "Failed to parse SwarmResult JSON: %s", err
                            )
                            return None
        except RedisError as exc:
            logger.error(
                "Redis error while listening for result on task %s: %s",
                task_id,
                exc,
            )
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.aclose()

        logger.warning(
            "Timed out waiting for result for task %s after %ss",
            task_id,
            timeout,
        )
        return None

    async def acknowledge_task(
        self, stream_key: str, group_name: str, message_id: str
    ) -> bool:
        """Acknowledge a processed stream message using XACK."""
        client = await self.get_client()
        try:
            acked_count = await client.xack(stream_key, group_name, message_id)
            return acked_count > 0
        except RedisError as exc:
            logger.error(
                "Failed to acknowledge message %s in stream %s: %s",
                message_id,
                stream_key,
                exc,
            )
            return False
=======
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
>>>>>>> c6f8ca0621e09e8e2ff4bedd0ad32bc0ddbc07f2
