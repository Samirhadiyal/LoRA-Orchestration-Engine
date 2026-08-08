# app/router/registry.py
from typing import Dict
from app.router.handlers import (
    RouteHandler,
    RetrievalHandler,
    ToolHandler,
    ExpertHandler,
    GenerationHandler
)
from app.router.exceptions import UnknownRouteAction

class HandlerRegistry:
    def __init__(self):
        # Maps string action names from the Planner to concrete Handlers
        self._handlers: Dict[str, RouteHandler] = {
            "retrieve": RetrievalHandler(),
            "tool": ToolHandler(),
            "expert": ExpertHandler(),
            "generate": GenerationHandler(),
        }

    def get_handler(self, action: str) -> RouteHandler:
        """Retrieves the appropriate handler for a given action. Raises an error if unknown."""
        handler = self._handlers.get(action)
        if not handler:
            raise UnknownRouteAction(f"No handler registered for action: '{action}'")
        return handler