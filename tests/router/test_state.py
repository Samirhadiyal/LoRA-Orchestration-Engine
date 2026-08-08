# tests/router/test_state.py
import uuid
from app.router.state import (
    ExecutionState,
    ExecutionStep,
    ExecutionStatus,
    StepStatus
)
from app.planner.schemas import Plan

def test_state_initializes_correctly():
    """Test: State initializes correctly with expected default values."""
    request_id = uuid.uuid4()
    dummy_plan = Plan()
    
    state = ExecutionState(
        request_id=request_id,
        user_query="Calculate revenue growth",
        plan=dummy_plan
    )
    
    assert state.request_id == request_id
    assert state.user_query == "Calculate revenue growth"
    assert state.status == ExecutionStatus.PENDING
    assert state.current_step_index == 0
    assert state.steps == []
    assert state.retrieved_context == []
    assert state.tool_results == []
    assert state.expert_output is None
    assert state.final_response is None
    assert state.citations == []
    assert state.error is None

def test_state_can_store_steps_and_results():
    """Test: Steps, retrieval results, tool results, and final response can be stored."""
    state = ExecutionState(
        request_id=uuid.uuid4(),
        user_query="What is the revenue?",
        plan=Plan()
    )
    
    step = ExecutionStep(step_id="step_1", action="retrieve")
    state.steps.append(step)
    assert len(state.steps) == 1
    assert state.steps[0].action == "retrieve"
    assert state.steps[0].status == StepStatus.PENDING
    
    state.retrieved_context.append({"doc_id": "123", "text": "Q1 Revenue was $5M"})
    assert len(state.retrieved_context) == 1
    assert state.retrieved_context[0]["doc_id"] == "123"
    
    state.tool_results.append({"tool": "calculator", "output": 5000000})
    assert len(state.tool_results) == 1
    
    state.final_response = "The Q1 revenue was $5 million."
    assert state.final_response == "The Q1 revenue was $5 million."

def test_state_failure_storage():
    """Test: Failure state can successfully store an error message."""
    state = ExecutionState(
        request_id=uuid.uuid4(),
        user_query="test",
        plan=Plan()
    )
    
    state.status = ExecutionStatus.FAILED
    state.error = "Unknown routing action requested."
    
    assert state.status == ExecutionStatus.FAILED
    assert state.error == "Unknown routing action requested."

def test_state_serialization():
    """Test: State serializes with Pydantic and can be perfectly recreated."""
    request_id = uuid.uuid4()
    original_state = ExecutionState(
        request_id=request_id,
        user_query="test serialization",
        plan=Plan(),
        status=ExecutionStatus.RUNNING,
        current_step_index=1,
        final_response="Success"
    )
    
    serialized_data = original_state.model_dump()
    
    assert serialized_data["user_query"] == "test serialization"
    assert serialized_data["status"] == "running"
    assert serialized_data["current_step_index"] == 1
    
    recreated_state = ExecutionState.model_validate(serialized_data)
    
    assert recreated_state.request_id == original_state.request_id
    assert recreated_state.status == ExecutionStatus.RUNNING
    assert recreated_state.final_response == "Success"