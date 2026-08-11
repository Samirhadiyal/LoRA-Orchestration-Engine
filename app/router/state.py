# app/router/state.py
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from app.planner.schemas import ExecutionPlan


class ExecutionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class StepStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class ExecutionStep(BaseModel):
    """Runtime representation of a planned step."""
    step_id: str
    action: str
    status: StepStatus = StepStatus.PENDING
    result: Any | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

class ExecutionState(BaseModel):
    """
    The pure-data state that travels through the execution pipeline.
    Must never contain infrastructure clients.
    """
    request_id: UUID
    session_id: UUID | None = None
    user_query: str
    
    plan: ExecutionPlan
    status: ExecutionStatus = ExecutionStatus.PENDING
    current_step_index: int = 0
    
    steps: list[ExecutionStep] = Field(default_factory=list)
    retrieved_context: str = ""
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    expert_output: str | None = None
    final_response: str | None = None
    citations: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None
