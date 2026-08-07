import json
import redis.asyncio as redis
from typing import List, Dict
from app.core.config import settings

# Gracefully read REDIS_URL from settings, or fall back to local Redis if not defined
REDIS_URL = getattr(settings, "REDIS_URL", "redis://localhost:6379/0")

class RedisSessionManager:
    def __init__(self):
        # We use async redis since FastAPI is asynchronous
        self.redis = redis.from_url(REDIS_URL, decode_responses=True)
        # Expiration time for chat history in Redis (24 hours)
        self.ttl = 86400 

    async def add_message(self, session_id: str, role: str, content: str):
        """Appends a new message to the session's chat history in Redis."""
        key = f"chat_history:{session_id}"
        message = json.dumps({"role": role, "content": content})
        
        # Push message to the end of the Redis list
        await self.redis.rpush(key, message)
        # Reset the expiration timer every time a new message is added
        await self.redis.expire(key, self.ttl)

    async def get_history(self, session_id: str) -> List[Dict]:
        """Retrieves the full chat history for a given session."""
        key = f"chat_history:{session_id}"
        messages_json = await self.redis.lrange(key, 0, -1)
        
        # Parse the JSON strings back into Python dictionaries
        return [json.loads(msg) for msg in messages_json]

    async def clear_history(self, session_id: str):
        """Deletes the session history from Redis."""
        key = f"chat_history:{session_id}"
        await self.redis.delete(key)

# Instantiate global instance
session_manager = RedisSessionManager()