# app/router/state.py
from enum import Enum
from typing import Any, List, Optional, Dict
from pydantic import BaseModel, Field
from uuid import UUID

from app.planner.schemas import ExecutionPlan

class ExecutionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class ExecutionStep(BaseModel):
    """Runtime representation of a planned step."""
    step_id: str
    action: str
    status: StepStatus = StepStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None

class ExecutionState(BaseModel):
    """
    The pure-data state that travels through the execution pipeline.
    Must never contain infrastructure clients.
    """
    request_id: UUID
    session_id: Optional[UUID] = None
    user_query: str
    
    plan: ExecutionPlan
    status: ExecutionStatus = ExecutionStatus.PENDING
    current_step_index: int = 0
    
    steps: List[ExecutionStep] = Field(default_factory=list)
    retrieved_context: List[Dict[str, Any]] = Field(default_factory=list)
    tool_results: List[Dict[str, Any]] = Field(default_factory=list)
    expert_output: Optional[str] = None
    final_response: Optional[str] = None
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = None