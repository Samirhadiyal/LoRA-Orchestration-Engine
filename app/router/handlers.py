# app/router/handlers.py
import logging
from typing import Protocol

from app.router.state import ExecutionState, ExecutionStep, StepStatus
from app.llm.lora_manager import LoRAManager
from app.llm.exceptions import AdapterNotFoundError, AdapterLoadError
from app.retrieval.search import hybrid_search  # Phase 3 Qdrant/BM25 integration

logger = logging.getLogger(__name__)

# Instantiate globally so the VRAM cache persists across API requests
lora_manager = LoRAManager()

class RouteHandler(Protocol):
    """Contract that all execution handlers must follow."""
    async def execute(self, state: ExecutionState, step: ExecutionStep) -> ExecutionState:
        ...

class RetrievalHandler:
    async def execute(self, state: ExecutionState, step: ExecutionStep) -> ExecutionState:
        logger.info(f"Executing RetrievalHandler for step {step.step_id}")
        step.status = StepStatus.IN_PROGRESS
        
        try:
            # 1. Execute actual hybrid search against Qdrant
            # (Assuming hybrid_search takes the query and returns a list of chunk dicts)
            search_results = hybrid_search(state.user_query)
            
            # 2. Format results into a single context string
            if not search_results:
                state.retrieved_context = "No relevant documents found in the knowledge base."
            else:
                formatted_context = "--- RETRIEVED CONTEXT ---\n"
                # Handle standard dictionary structures from the Phase 3 search output
                for i, res in enumerate(search_results):
                    content = res.get("text", res.get("content", str(res)))
                    formatted_context += f"[Source {i+1}]: {content}\n\n"
                
                state.retrieved_context = formatted_context

            step.status = StepStatus.COMPLETED
            step.result = {"status": "success", "chunks_retrieved": len(search_results)}
            
        except Exception as e:
            logger.error(f"RAG Retrieval failed: {e}")
            step.status = StepStatus.FAILED
            step.result = {"status": "error"}
            state.error = f"Retrieval failed: {str(e)}"
            
        return state

class ToolHandler:
    async def execute(self, state: ExecutionState, step: ExecutionStep) -> ExecutionState:
        # DO NOT TOUCH: Engineer A is actively working on this class!
        state.tool_results[step.step_id] = {"tool": "mock_calculator", "output": "Mock tool result."}
        step.status = StepStatus.COMPLETED
        step.result = "Tool executed successfully"
        return state

class ExpertHandler:
    async def execute(self, state: ExecutionState, step: ExecutionStep) -> ExecutionState:
        # Fixed signature to match RouteHandler protocol
        logger.info(f"Executing ExpertHandler for step {step.step_id}")
        step.status = StepStatus.IN_PROGRESS
        
        # 1. Extract the requested expert from the AI's Plan
        suggested_lora = state.plan.suggested_lora
        
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
            
        except Exception as e:
            logger.error(f"Unexpected system crash in ExpertHandler: {e}")
            step.status = StepStatus.FAILED
            step.result = {"status": "error", "error_type": "Unexpected"}
            state.error = "Critical failure during expert routing."
            
        return state

class GenerationHandler:
    async def execute(self, state: ExecutionState, step: ExecutionStep) -> ExecutionState:
        # Mock implementation for Phase 3 routing contract (Will be updated in Step B5)
        state.final_response = "This is a mock final response based on retrieved data and tools."
        step.status = StepStatus.COMPLETED
        step.result = "Generation successful"
        return state