# app/router/graph.py
from langgraph.graph import StateGraph, END
from app.router.state import ExecutionState, ExecutionStatus, StepStatus
from app.router.registry import HandlerRegistry

# Initialize our registry from B2
registry = HandlerRegistry()

async def execute_step(state: ExecutionState) -> ExecutionState:
    """
    Core graph node: Retrieves the current step, finds the correct handler, 
    executes it, and updates the state.
    """
    if state.status == ExecutionStatus.FAILED or state.current_step_index >= len(state.steps):
        return state
        
    # Get the current step based on the index
    current_step = state.steps[state.current_step_index]
    current_step.status = StepStatus.RUNNING
    
    try:
        # Ask the registry for the correct handler (e.g., RetrievalHandler, GenerationHandler)
        handler = registry.get_handler(current_step.action)
        
        # Execute the handler and update the state
        state = await handler.execute(state, current_step)

        if current_step.status == StepStatus.FAILED:
            state.status = ExecutionStatus.FAILED
            state.error = current_step.error or f"Execution failed on step '{current_step.step_id}'"
        else:
            current_step.status = StepStatus.COMPLETED
            state.current_step_index += 1
            if state.current_step_index >= len(state.steps):
                state.status = ExecutionStatus.COMPLETED
        
    except Exception as e:
        # If anything fails safely catch it and mark the execution as failed
        current_step.status = StepStatus.FAILED
        current_step.error = str(e)
        state.status = ExecutionStatus.FAILED
        state.error = f"Execution failed on step '{current_step.step_id}': {str(e)}"
        
    return state

def check_next_action(state: ExecutionState) -> str:
    """
    Conditional edge: Decides if the graph should loop back to execute 
    another step or finish the execution.
    """
    if state.status == ExecutionStatus.FAILED:
        return END
        
    # If there are more steps in the array, loop back
    if state.current_step_index < len(state.steps):
        return "execute_step"
        
    return END

# --- Compile the LangGraph ---
workflow = StateGraph(ExecutionState)

# Add our single powerful node
workflow.add_node("execute_step", execute_step)
workflow.set_entry_point("execute_step")

# Add the conditional routing logic
workflow.add_conditional_edges(
    "execute_step",
    check_next_action,
    {
        "execute_step": "execute_step",
        END: END
    }
)

# Export the compiled graph
pipeline_graph = workflow.compile()
