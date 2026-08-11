import logging
import time

import redis.asyncio as aioredis
from pydantic import ValidationError
from redis.exceptions import RedisError

from app.swarm.schemas import SwarmResult, SwarmTask

logger = logging.getLogger(__name__)


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