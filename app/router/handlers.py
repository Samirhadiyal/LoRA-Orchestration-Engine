# app/router/handlers.py
from typing import Protocol
from app.router.state import ExecutionState, ExecutionStep, StepStatus

class RouteHandler(Protocol):
    """Contract that all execution handlers must follow."""
    async def execute(self, state: ExecutionState, step: ExecutionStep) -> ExecutionState:
        ...

class RetrievalHandler:
    async def execute(self, state: ExecutionState, step: ExecutionStep) -> ExecutionState:
        # Mock implementation for Phase 3 routing contract
        state.retrieved_context.append({"source": "knowledge_base", "text": "Mock retrieved data."})
        step.status = StepStatus.COMPLETED
        step.result = "Retrieval successful"
        return state

class ToolHandler:
    async def execute(self, state: ExecutionState, step: ExecutionStep) -> ExecutionState:
        # Mock implementation for Phase 3 routing contract
        state.tool_results.append({"tool": "mock_calculator", "output": "Mock tool result."})
        step.status = StepStatus.COMPLETED
        step.result = "Tool executed successfully"
        return state

class ExpertHandler:
    async def execute(self, state: ExecutionState, step: ExecutionStep) -> ExecutionState:
        # Mock implementation for Phase 3 routing contract
        state.expert_output = "Mock expert/LoRA output."
        step.status = StepStatus.COMPLETED
        step.result = "Expert analysis complete"
        return state

class GenerationHandler:
    async def execute(self, state: ExecutionState, step: ExecutionStep) -> ExecutionState:
        # Mock implementation for Phase 3 routing contract
        state.final_response = "This is a mock final response based on retrieved data and tools."
        step.status = StepStatus.COMPLETED
        step.result = "Generation successful"
        return state