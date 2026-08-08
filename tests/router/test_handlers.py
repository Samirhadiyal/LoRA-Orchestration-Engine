# tests/router/test_handlers.py
import pytest
import uuid
from app.router.state import ExecutionState, ExecutionStep, StepStatus
from app.planner.schemas import Plan
from app.router.registry import HandlerRegistry
from app.router.exceptions import UnknownRouteAction

@pytest.fixture
def base_state():
    return ExecutionState(
        request_id=uuid.uuid4(),
        user_query="Test query",
        plan=Plan()
    )

def test_registry_routing_valid_and_invalid():
    """Test: Registry correctly routes valid actions and securely blocks invalid ones."""
    registry = HandlerRegistry()
    
    assert registry.get_handler("retrieve") is not None
    assert registry.get_handler("tool") is not None
    
    # Simulate a malformed or malicious plan
    with pytest.raises(UnknownRouteAction):
        registry.get_handler("destroy_database")

@pytest.mark.asyncio
async def test_retrieval_handler_execution(base_state):
    """Test: RetrievalHandler successfully updates state and step status."""
    registry = HandlerRegistry()
    handler = registry.get_handler("retrieve")
    step = ExecutionStep(step_id="step_1", action="retrieve")
    
    updated_state = await handler.execute(base_state, step)
    
    assert step.status == StepStatus.COMPLETED
    assert len(updated_state.retrieved_context) > 0
    assert updated_state.retrieved_context[0]["source"] == "knowledge_base"

@pytest.mark.asyncio
async def test_generation_handler_execution(base_state):
    """Test: GenerationHandler successfully populates the final response."""
    registry = HandlerRegistry()
    handler = registry.get_handler("generate")
    step = ExecutionStep(step_id="step_2", action="generate")
    
    updated_state = await handler.execute(base_state, step)
    
    assert step.status == StepStatus.COMPLETED
    assert updated_state.final_response is not None