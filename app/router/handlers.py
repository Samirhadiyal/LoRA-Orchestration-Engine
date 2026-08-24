import logging
import os
import uuid
from typing import Any, Protocol

from app.llm.exceptions import AdapterLoadError, AdapterNotFoundError
from app.llm.lora_manager import LoRAManager
from app.retrieval.search import hybrid_search  # Phase 3 Qdrant/BM25 integration
from app.router.state import ExecutionState, ExecutionStep, StepStatus
from app.tools.mcp_client import MCPClient
from app.tools.sql_agent import SQLAgent

logger = logging.getLogger(__name__)

# Instantiate globally so the VRAM cache persists across API requests
lora_manager = LoRAManager()


class SwarmHandler:
    """
    Phase 6B Track B: Swarm Delegation Handler.

    Delegates step execution to background specialized swarm workers via SwarmMessageBus.
    Adheres strictly to the immutable RouteHandler protocol:
        async def execute(self, state: ExecutionState, step: ExecutionStep) -> ExecutionState
    """

    def __init__(self, message_bus: Any = None, default_timeout_seconds: float = 30.0):
        self.message_bus = message_bus
        self.default_timeout_seconds = default_timeout_seconds

    def _is_swarm_enabled(self) -> bool:
        """Checks environment override NEUROMESH_SWARM_ENABLED."""
        env_val = os.getenv("NEUROMESH_SWARM_ENABLED", "").strip().lower()
        return env_val in ("true", "1", "yes")

    async def execute(self, state: ExecutionState, step: ExecutionStep) -> ExecutionState:
        """
        Dispatches minimal task payload to the SwarmMessageBus and awaits worker result.
        Never sends the full ExecutionState over the wire.
        """
        if not self._is_swarm_enabled() or self.message_bus is None:
            logger.warning(
                "SwarmHandler invoked while swarm is disabled or message_bus is uninitialized (step=%s).",
                step.step_id,
            )
            step.status = StepStatus.FAILED
            state.error = "Swarm execution disabled or uninitialized."
            return state

        task_id = str(uuid.uuid4())
        step.status = StepStatus.RUNNING
        logger.info("Dispatching step '%s' to Swarm (task_id=%s)", step.step_id, task_id)

        try:
            # 1. Dispatch minimal task payload (never serialize full state)
            task_payload = {
                "task_id": task_id,
                "step_id": step.step_id,
                "action": step.action,
                "metadata": getattr(step, "metadata", {}) or {},
            }
            await self.message_bus.publish_task(worker_id="swarm-pool", payload=task_payload)

            # 2. Await worker result from message bus with timeout
            result_data = await self.message_bus.listen_for_result(
                task_id=task_id, timeout=self.default_timeout_seconds
            )

            if not result_data or not result_data.get("success", False):
                err_msg = (
                    result_data.get("error")
                    if result_data
                    else "Swarm worker timed out or returned no result."
                )
                logger.error("Swarm execution failed for step '%s': %s", step.step_id, err_msg)
                step.status = StepStatus.FAILED
                state.error = err_msg
                return state

            # 3. Apply worker output back to canonical state
            worker_output = result_data.get("result", {})
            step.result = worker_output
            step.status = StepStatus.COMPLETED

            # Store in tool_results so ContextOptimizer V2 processes it
            if not hasattr(state, "tool_results") or state.tool_results is None:
                state.tool_results = []
            state.tool_results.append({step.step_id: worker_output})

            logger.info("Swarm step '%s' completed successfully.", step.step_id)
            return state

        except (RuntimeError, TimeoutError, OSError, ValueError) as exc:
            logger.error("Error during SwarmHandler execution for step '%s': %s", step.step_id, exc)
            step.status = StepStatus.FAILED
            state.error = str(exc)
            return state


class RouteHandler(Protocol):
    """Contract that all execution handlers must follow."""
    async def execute(self, state: ExecutionState, step: ExecutionStep) -> ExecutionState:
        ...


class RetrievalHandler:
    async def execute(self, state: ExecutionState, step: ExecutionStep) -> ExecutionState:
        logger.info(f"Executing RetrievalHandler for step {step.step_id}")
        step.status = StepStatus.IN_PROGRESS
        
        try:
            search_results = hybrid_search(state.user_query)
            
            if not search_results:
                state.retrieved_context = "No relevant documents found in the knowledge base."
            else:
                formatted_context = "--- RETRIEVED CONTEXT ---\n"
                for i, res in enumerate(search_results):
                    content = res.get("text", res.get("content", str(res)))
                    formatted_context += f"[Source {i+1}]: {content}\n\n"
                
                state.retrieved_context = formatted_context

            step.status = StepStatus.COMPLETED
            step.result = {"status": "success", "chunks_retrieved": len(search_results)}
            
        except (ConnectionError, RuntimeError) as e:
            logger.warning("RAG retrieval unavailable; continuing with empty context: %s", e)
            state.retrieved_context = (
                "No external knowledge base context available "
                "(Qdrant collection uninitialized or offline)."
            )
            step.status = StepStatus.COMPLETED
            step.result = {"status": "fallback_empty_context", "message": str(e)}
            
        return state


class ToolHandler:
    """
    Handles execution steps that require external tools or database queries.
    """

    def __init__(self):
        self.sql_agent = SQLAgent()
        self.mcp_client = MCPClient()

    async def execute(self, state: ExecutionState, step: ExecutionStep) -> ExecutionState:
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
            step.status = StepStatus.COMPLETED

        return state


class ExpertHandler:
    async def execute(self, state: ExecutionState, step: ExecutionStep) -> ExecutionState:
        logger.info(f"Executing ExpertHandler for step {step.step_id}")
        step.status = StepStatus.IN_PROGRESS
        
        # 1. Extract the requested expert from the AI's Plan
        suggested_lora = state.plan.suggested_lora if state.plan else None
        
        if not suggested_lora:
            logger.info("No suggested_lora provided in the plan. Defaulting to base model.")
            step.status = StepStatus.COMPLETED
            step.result = {"status": "skipped", "message": "No LoRA requested"}
            return state
            
        # 2. Safely attempt to load the adapter
        try:
            success = await lora_manager.load_adapter(suggested_lora)
            if success:
                step.status = StepStatus.COMPLETED
                step.result = {"status": "success", "adapter_id": suggested_lora}
            else:
                step.status = StepStatus.FAILED
                state.error = f"Failed to load adapter: {suggested_lora}"
                
        # 3. Catch explicit domain exceptions
        except AdapterNotFoundError as e:
            logger.error(f"Routing Error: {e}")
            step.status = StepStatus.FAILED
            step.result = {"status": "error", "error_type": "AdapterNotFoundError"}
            state.error = str(e)
            
        except AdapterLoadError as e:
            logger.error(f"Hardware Error: {e}")
            step.status = StepStatus.FAILED
            step.result = {"status": "error", "error_type": "AdapterLoadError"}
            state.error = str(e)
            
        except (RuntimeError, OSError) as e:
            logger.error("Unexpected system crash in ExpertHandler: %s", e)
            step.status = StepStatus.FAILED
            step.result = {"status": "error", "error_type": "Unexpected"}
            state.error = "Critical failure during expert routing."
            
        return state


class GenerationHandler:
    async def execute(self, state: ExecutionState, step: ExecutionStep) -> ExecutionState:
        state.final_response = "This is a mock final response based on retrieved data and tools."
        step.status = StepStatus.COMPLETED
        step.result = "Generation successful"
        return state


class SwarmHandler:
    """
    Phase 6B Track B: Swarm Delegation Handler.

    Delegates step execution to background specialized swarm workers via SwarmMessageBus.
    Adheres strictly to the immutable RouteHandler protocol:
        async def execute(self, state: ExecutionState, step: ExecutionStep) -> ExecutionState
    """

    def __init__(self, message_bus: Any = None, default_timeout_seconds: float = 30.0):
        self.message_bus = message_bus
        self.default_timeout_seconds = default_timeout_seconds

    def _is_swarm_enabled(self) -> bool:
        """Checks environment override NEUROMESH_SWARM_ENABLED."""
        env_val = os.getenv("NEUROMESH_SWARM_ENABLED", "").strip().lower()
        return env_val in ("true", "1", "yes")

    async def execute(self, state: ExecutionState, step: ExecutionStep) -> ExecutionState:
        """
        Dispatches minimal task payload to the SwarmMessageBus and awaits worker result.
        Never sends the full ExecutionState over the wire.
        """
        if not self._is_swarm_enabled() or self.message_bus is None:
            logger.warning(
                "SwarmHandler invoked while swarm is disabled or message_bus is uninitialized (step=%s).",
                step.step_id,
            )
            step.status = StepStatus.FAILED
            state.error = "Swarm execution disabled or uninitialized."
            return state

        task_id = str(uuid.uuid4())
        step.status = StepStatus.RUNNING
        logger.info("Dispatching step '%s' to Swarm (task_id=%s)", step.step_id, task_id)

        try:
            task_payload = {
                "task_id": task_id,
                "step_id": step.step_id,
                "action": step.action,
                "metadata": getattr(step, "metadata", {}) or {},
            }
            await self.message_bus.publish_task(worker_id="swarm-pool", payload=task_payload)

            result_data = await self.message_bus.listen_for_result(
                task_id=task_id, timeout=self.default_timeout_seconds
            )

            if not result_data or not result_data.get("success", False):
                err_msg = (
                    result_data.get("error")
                    if result_data
                    else "Swarm worker timed out or returned no result."
                )
                logger.error("Swarm execution failed for step '%s': %s", step.step_id, err_msg)
                step.status = StepStatus.FAILED
                state.error = err_msg
                return state

            worker_output = result_data.get("result", {})
            step.result = worker_output
            step.status = StepStatus.COMPLETED

            # Robust state update: handle both dict and list schemas for tool_results
            if getattr(state, "tool_results", None) is None:
                state.tool_results = {}
            
            if isinstance(state.tool_results, list):
                state.tool_results.append({step.step_id: worker_output})
            elif isinstance(state.tool_results, dict):
                state.tool_results[step.step_id] = worker_output

            logger.info("Swarm step '%s' completed successfully.", step.step_id)
            return state

        except (RuntimeError, TimeoutError, OSError, ValueError) as exc:
            logger.error("Error during SwarmHandler execution for step '%s': %s", step.step_id, exc)
            step.status = StepStatus.FAILED
            state.error = str(exc)
            return state
