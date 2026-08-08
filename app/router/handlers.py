# app/router/handlers.py
from typing import Protocol
from app.router.state import ExecutionState, ExecutionStep, StepStatus
from app.tools.sql_agent import SQLAgent
from app.tools.mcp_client import MCPClient

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
    """
    Handles execution steps that require external tools or database queries.
    """

    def __init__(self):
        self.sql_agent = SQLAgent()
        self.mcp_client = MCPClient()

    async def execute(self, state, step) -> None:
        """
        Routes the step to SQLAgent or MCPClient based on tool type,
        stores results in state.tool_results, and updates step status.
        """
        tool = getattr(step, "tool", getattr(step, "action", "")).lower()
        query_input = getattr(step, "query_input", getattr(step, "description", ""))
        step_id = getattr(step, "step_id", getattr(step, "step_number", 1))

        # 1. Route based on tool type
        if "sql" in tool:
            result = await self.sql_agent.execute_query(query_input)
        else:
            result = await self.mcp_client.fetch_external_data(tool_name=tool, query=query_input)

        # 2. Store result in state dictionary
        if not hasattr(state, "tool_results") or state.tool_results is None:
            state.tool_results = {}

        state.tool_results[step_id] = result

        # 3. Update step status to COMPLETED
        if hasattr(step, "status"):
            from app.router.state import StepStatus
            step.status = StepStatus.COMPLETED

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