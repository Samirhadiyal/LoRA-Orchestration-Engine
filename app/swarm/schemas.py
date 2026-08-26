<<<<<<< HEAD
=======
# Path: app/swarm/schemas.py
>>>>>>> c6f8ca0621e09e8e2ff4bedd0ad32bc0ddbc07f2
from typing import Any

from pydantic import BaseModel, Field


class SwarmTask(BaseModel):
<<<<<<< HEAD
    task_id: str
    execution_id: str
    step_id: str
    required_capability: str
    payload: dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: float = 30.0
    attempt: int = 1


class SwarmResult(BaseModel):
    task_id: str
    execution_id: str
=======
    """Network payload dispatched to a Swarm Worker."""
    task_id: str
    step_id: str
    action: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    run_id: str | None = None


class SwarmResult(BaseModel):
    """Network payload returned by a Swarm Worker."""
    task_id: str
>>>>>>> c6f8ca0621e09e8e2ff4bedd0ad32bc0ddbc07f2
    step_id: str
    worker_id: str
    success: bool
    result: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
<<<<<<< HEAD
    execution_time_seconds: float
=======
    execution_time_seconds: float | None = None


class WorkerHeartbeat(BaseModel):
    """Ephemeral health state broadcasted by Workers."""
    worker_id: str
    capabilities: set[str]
    load: float
    status: str
>>>>>>> c6f8ca0621e09e8e2ff4bedd0ad32bc0ddbc07f2
