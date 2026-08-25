import uuid
from typing import Any
from langsmith.run_helpers import get_current_run_tree

from app.planner.analyzer import TaskAnalyzer  # Engineer A's code!
from app.router.graph import pipeline_graph
from app.router.state import ExecutionState, ExecutionStep


class OrchestrationService:
    @staticmethod
    async def process_chat(user_query: str, session_id: str | None = None) -> dict[str, Any]:
        """
        Orchestrates the entire flow:
        1. Calls the Task Analyzer (Planner) to get a real Plan from the LLM.
        2. Initializes the Execution State.
        3. Runs the LangGraph execution engine.
        """
        
        # 1. Ask the AI to generate the real plan!
        analyzer = TaskAnalyzer()
        real_plan = await analyzer.analyze_query(user_query)
        
        # Capture current run ID from LangSmith for distributed tracing
        current_run = get_current_run_tree()
        run_id = str(current_run.id) if current_run else str(uuid.uuid4())
        
        # 2. Map Engineer A's "TaskStep" into Engineer B's "ExecutionStep"
        execution_steps = [
            ExecutionStep(
                step_id=f"step_{step.step_number}", 
                action=step.tool,
                metadata={
                    "worker_capability": getattr(step, "worker_capability", None),
                    "run_id": run_id
                }
            )
            for step in real_plan.steps
        ]
        
        # 3. Initialize the state
        initial_state = ExecutionState(
            request_id=uuid.uuid4(),
            session_id=uuid.UUID(session_id) if session_id else None,
            user_query=user_query,
            plan=real_plan,
            steps=execution_steps
        )
        
        # 4. Execute the LangGraph workflow asynchronously
        final_state_data = await pipeline_graph.ainvoke(initial_state)
        
        # LangGraph returns a dictionary or the updated BaseModel depending on the version.
        if isinstance(final_state_data, dict):
            final_state = ExecutionState.model_validate(final_state_data)
        else:
            final_state = final_state_data
        
        # 5. Format the final API response
        return {
            "request_id": str(final_state.request_id),
            "status": final_state.status.value,
            "answer": final_state.final_response or "No response generated.",
            "citations": final_state.citations,
            "error": final_state.error,
            "execution_steps": [
                {"step_id": step.step_id, "action": step.action, "status": step.status.value} 
                for step in final_state.steps
            ]
        }