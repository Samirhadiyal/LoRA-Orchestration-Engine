from typing import Any

from pydantic import BaseModel, Field


class SwarmTask(BaseModel):
    """Network payload dispatched to a Swarm Worker."""

    task_id: str
    execution_id: str
    step_id: str
    required_capability: str
    payload: dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: float = 30.0
    attempt: int = 1


class SwarmResult(BaseModel):
    """Network payload returned by a Swarm Worker."""

    task_id: str
    execution_id: str
    step_id: str
    worker_id: str
    success: bool
    result: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    execution_time_seconds: float = 0.0


class WorkerHeartbeat(BaseModel):
    """Heartbeat payload sent by Swarm Workers."""

    worker_id: str
    capabilities: list[str]
    current_load: float = 0.0
