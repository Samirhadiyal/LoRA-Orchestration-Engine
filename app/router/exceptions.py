# app/router/exceptions.py

class RouterError(Exception):
    """Base exception for the pipeline router."""
    pass

class UnknownRouteAction(RouterError):
    """Raised when the planner requests an action not in the registry."""
    pass

class HandlerExecutionError(RouterError):
    """Raised when a specific handler fails to execute."""
    pass

class ExecutionStateError(RouterError):
    """Raised when there is an invalid state transition."""
    pass

class PlanExecutionError(RouterError):
    """Raised when the overall plan fails."""
    pass