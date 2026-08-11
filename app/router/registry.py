from app.router.handlers import (
    ExpertHandler,
    GenerationHandler,
    RetrievalHandler,
    ToolHandler,
)


class HandlerRegistry:
    def __init__(self):
        # This maps the Planner's tool requests to the actual Python classes
        self._handlers = {
            "rag_search": RetrievalHandler(),
            "sql_query": ToolHandler(),
            "lora_adapter": ExpertHandler(),
            "direct_llm": GenerationHandler(),
        }

    def get_handler(self, action: str):
        return self._handlers.get(action)

# Global singleton for the router to use
handler_registry = HandlerRegistry()
