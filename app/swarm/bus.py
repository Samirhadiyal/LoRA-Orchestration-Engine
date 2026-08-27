import asyncio
import json
import logging
from typing import Any, cast

import redis.asyncio as redis
from pydantic import ValidationError
from redis.exceptions import RedisError

from app.swarm.schemas import SwarmResult, SwarmTask

logger = logging.getLogger(__name__)


class SwarmMessageBus:
    """Async Redis Streams & Pub/Sub transport layer for Swarm orchestration."""

    def __init__(self, redis_url: str = "redis://localhost:6379/0") -> None:
        self.redis_url = redis_url
        self._client: redis.Redis | None = None

    async def get_client(self) -> redis.Redis:
        """Lazily initialize and return the async Redis connection."""
        if self._client is None:
            self._client = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=False,
            )
        return self._client

    async def publish_task(
        self,
        stream_key: str | None = None,
        task: SwarmTask | None = None,
        *,
        worker_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> str:
        """Publishes a task payload into a Redis Stream via XADD.
        Supports both direct SwarmTask object and worker_id/payload keyword invocation."""
        client = await self.get_client()

        target_stream = stream_key or f"swarm:tasks:{worker_id}"

        if task is not None:
            task_obj = task
        elif payload is not None:
            await client.xadd(target_stream, {"payload": json.dumps(payload)})  # type: ignore[arg-type]
            return str(payload.get("task_id", ""))
        else:
            raise ValueError("Either 'task' or 'payload' must be provided to publish_task")

        message_data: dict[str, str] = {
            "task_id": task_obj.task_id,
            "execution_id": task_obj.execution_id,
            "step_id": task_obj.step_id,
            "required_capability": task_obj.required_capability,
            "payload": task_obj.model_dump_json(),
        }
        msg_id = await client.xadd(target_stream, message_data)  # type: ignore[arg-type]
        return msg_id.decode("utf-8") if isinstance(msg_id, bytes) else str(msg_id)

    async def publish_result(self, result: SwarmResult | dict[str, Any]) -> int | bool:
        """Publishes execution result to a unique task result Pub/Sub channel."""
        client = await self.get_client()
        if isinstance(result, dict):
            task_id = str(result.get("task_id", ""))
            subscribers: int = await client.publish(
                f"swarm:results:{task_id}", json.dumps(result)
            )
            return subscribers > 0

        channel = f"swarm:results:{result.task_id}"
        return await client.publish(channel, result.model_dump_json())

    async def acknowledge_task(
        self, stream_key: str, group_name: str, message_id: str
    ) -> bool:
        """Acknowledges task processing completion via XACK."""
        client = await self.get_client()
        try:
            ack_count: int = await client.xack(stream_key, group_name, message_id)
            return ack_count > 0
        except RedisError as e:
            logger.error(
                "Failed to acknowledge message %s in stream %s: %s",
                message_id,
                stream_key,
                e,
            )
            return False

    async def listen_for_task(self, worker_id: str, timeout: float = 1.0) -> dict[str, Any] | None:
        """Polls the worker's dedicated task stream for new tasks."""
        client = await self.get_client()
        stream_key = f"swarm:tasks:{worker_id}"
        try:
            messages = cast(
                list[tuple[Any, list[tuple[Any, dict[Any, Any]]]]],
                await client.xread({stream_key: "0-0"}, count=1, block=int(timeout * 1000)),
            )
            if messages:
                for _, stream_messages in messages:
                    for _, data in stream_messages:
                        payload_raw = data.get(b"payload") or data.get("payload")
                        if isinstance(payload_raw, bytes):
                            payload_raw = payload_raw.decode("utf-8")
                        return json.loads(str(payload_raw))
        except Exception as e:
            logger.debug("Error listening for task on %s: %s", stream_key, e)
        return None

    async def listen_for_result(
        self, task_id: str, timeout: float = 30.0
    ) -> SwarmResult | None:
        """Subscribes to a task's unique Pub/Sub result channel and waits for output."""
        client = await self.get_client()
        pubsub = client.pubsub()
        channel = f"swarm:results:{task_id}"

        await pubsub.subscribe(channel)

        try:
            async with asyncio.timeout(timeout):
                while True:
                    message: dict[str, Any] | None = await pubsub.get_message(
                        ignore_subscribe_messages=True, timeout=0.1
                    )
                    if message and message.get("type") == "message":
                        raw_data = message.get("data")
                        json_str: str = (
                            raw_data.decode("utf-8")
                            if isinstance(raw_data, (bytes, bytearray))
                            else str(raw_data or "")
                        )
                        try:
                            return SwarmResult.model_validate_json(json_str)
                        except ValidationError as e:
                            logger.error("Failed to parse SwarmResult payload: %s", e)
                            return None
                    await asyncio.sleep(0.01)
        except asyncio.TimeoutError:
            logger.warning("Timed out waiting for task %s result on channel %s", task_id, channel)
            return None
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.aclose()

    async def close(self) -> None:
        """Closes the Redis connection."""
        if self._client:
            await self._client.aclose()
            self._client = None


RedisSwarmBus = SwarmMessageBus
