# Path: app/swarm/schemas.py
from typing import Any

from pydantic import BaseModel, Field


class SwarmTask(BaseModel):
    """Network payload dispatched to a Swarm Worker."""
    task_id: str
    step_id: str
    action: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    run_id: str | None = None


class SwarmResult(BaseModel):
    """Network payload returned by a Swarm Worker."""
    task_id: str
    step_id: str
    worker_id: str
    success: bool
    result: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    execution_time_seconds: float | None = None


class WorkerHeartbeat(BaseModel):
    """Ephemeral health state broadcasted by Workers."""
    worker_id: str
    capabilities: set[str]
    load: float
    status: str
