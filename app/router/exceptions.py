# app/router/exceptions.py

class RouterError(Exception):
    """Base exception for the pipeline router."""

class UnknownRouteAction(RouterError):
    """Raised when the planner requests an action not in the registry."""

class HandlerExecutionError(RouterError):
    """Raised when a specific handler fails to execute."""

class ExecutionStateError(RouterError):
    """Raised when there is an invalid state transition."""

class PlanExecutionError(RouterError):
    """Raised when the overall plan fails."""
