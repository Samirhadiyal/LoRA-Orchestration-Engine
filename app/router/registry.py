import os

from app.router.handlers import (
    ExpertHandler,
    GenerationHandler,
    RetrievalHandler,
    SwarmHandler,
    ToolHandler,
)
from app.swarm.bus import RedisSwarmBus


class HandlerRegistry:
    def __init__(self):
        # This maps the Planner's tool requests to the actual Python classes
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self._handlers = {
            "rag_search": RetrievalHandler(),
            "sql_query": ToolHandler(),
            "lora_adapter": ExpertHandler(),
            "direct_llm": GenerationHandler(),
            "swarm_delegate": SwarmHandler(message_bus=RedisSwarmBus(redis_url=redis_url)),
        }

    def get_handler(self, action: str):
        return self._handlers.get(action)

# Global singleton for the router to use
handler_registry = HandlerRegistry()
