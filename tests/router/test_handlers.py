# tests/router/test_handlers.py
import pytest
import uuid
from app.router.state import ExecutionState, ExecutionStep, StepStatus
from app.planner.schemas import ExecutionPlan
from app.router.registry import HandlerRegistry


def make_plan():
    return ExecutionPlan(user_intent="test", requires_retrieval=False, steps=[])

@pytest.fixture
def base_state():
    return ExecutionState(
        request_id=uuid.uuid4(),
        user_query="Test query",
        plan=make_plan()
    )

def test_registry_routing_valid_and_invalid():
    """Test: Registry correctly routes valid actions and securely blocks invalid ones."""
    registry = HandlerRegistry()
    
    assert registry.get_handler("rag_search") is not None
    assert registry.get_handler("direct_llm") is not None
    
    assert registry.get_handler("destroy_database") is None

@pytest.mark.asyncio
async def test_retrieval_handler_execution(base_state, monkeypatch):
    """Test: RetrievalHandler successfully updates state and step status."""
    monkeypatch.setattr(
        "app.router.handlers.hybrid_search",
        lambda query: [{"text": "Q1 Revenue was $5M", "doc_id": "doc_1"}],
    )
    registry = HandlerRegistry()
    handler = registry.get_handler("rag_search")
    step = ExecutionStep(step_id="step_1", action="rag_search")
    
    updated_state = await handler.execute(base_state, step)
    
    assert step.status == StepStatus.COMPLETED
    assert "--- RETRIEVED CONTEXT ---" in updated_state.retrieved_context
    assert "Q1 Revenue was $5M" in updated_state.retrieved_context
    assert step.result == {"status": "success", "chunks_retrieved": 1}

@pytest.mark.asyncio
async def test_retrieval_handler_falls_back_on_search_error(base_state, monkeypatch):
    """Test: RetrievalHandler does not fail the graph when Qdrant is unavailable."""
    def raise_missing_collection(query):
        raise RuntimeError("Collection 'neuromesh_knowledge' doesn't exist")

    monkeypatch.setattr("app.router.handlers.hybrid_search", raise_missing_collection)
    registry = HandlerRegistry()
    handler = registry.get_handler("rag_search")
    step = ExecutionStep(step_id="step_1", action="rag_search")

    updated_state = await handler.execute(base_state, step)

    assert step.status == StepStatus.COMPLETED
    assert updated_state.error is None
    assert updated_state.retrieved_context == (
        "No external knowledge base context available "
        "(Qdrant collection uninitialized or offline)."
    )
    assert step.result == {
        "status": "fallback_empty_context",
        "message": "Collection 'neuromesh_knowledge' doesn't exist",
    }

@pytest.mark.asyncio
async def test_generation_handler_execution(base_state):
    """Test: GenerationHandler successfully populates the final response."""
    registry = HandlerRegistry()
    handler = registry.get_handler("direct_llm")
    step = ExecutionStep(step_id="step_2", action="direct_llm")
    
    updated_state = await handler.execute(base_state, step)
    
    assert step.status == StepStatus.COMPLETED
    assert updated_state.final_response is not None
