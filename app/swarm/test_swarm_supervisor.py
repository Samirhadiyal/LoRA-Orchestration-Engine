# Path: tests/swarm/test_swarm_supervisor.py
import pytest

from app.router.state import ExecutionStep, StepStatus
from app.swarm.registry import WorkerProfile, WorkerRegistry
from app.swarm.supervisor import SwarmSupervisorPolicy


@pytest.mark.asyncio
async def test_swarm_disabled_default_routes_local(monkeypatch):
    """
    Verifies Backward Compatibility Requirement:
    When NEUROMESH_SWARM_ENABLED is false/unset, all tasks route LOCAL.
    """
    monkeypatch.setenv("NEUROMESH_SWARM_ENABLED", "false")
    policy = SwarmSupervisorPolicy()

    lora_step = ExecutionStep(
        step_id="step-1",
        action="lora_adapter",
        status=StepStatus.PENDING,
        metadata={"adapter_id": "finance_v1"},
    )

    route = policy.decide_execution_route(lora_step)
    assert route == "LOCAL", f"Expected LOCAL when swarm is disabled, got {route}"


@pytest.mark.asyncio
async def test_swarm_enabled_routes_heavy_tasks_to_swarm(monkeypatch):
    """
    Verifies Swarm Delegation Policy:
    When enabled, LoRA adapters, external MCP tools, and heavy RAG route to SWARM.
    """
    monkeypatch.setenv("NEUROMESH_SWARM_ENABLED", "true")
    policy = SwarmSupervisorPolicy()

    lora_step = ExecutionStep(
        step_id="step-1", action="lora_adapter", status=StepStatus.PENDING
    )
    mcp_step = ExecutionStep(
        step_id="step-2", action="mcp_tool", status=StepStatus.PENDING
    )
    heavy_rag_step = ExecutionStep(
        step_id="step-3",
        action="rag_search",
        status=StepStatus.PENDING,
        metadata={"heavy_rag": True, "top_k": 20},
    )
    light_rag_step = ExecutionStep(
        step_id="step-4",
        action="rag_search",
        status=StepStatus.PENDING,
        metadata={"top_k": 3},
    )

    assert policy.decide_execution_route(lora_step) == "SWARM"
    assert policy.decide_execution_route(mcp_step) == "SWARM"
    assert policy.decide_execution_route(heavy_rag_step) == "SWARM"
    assert policy.decide_execution_route(light_rag_step) == "LOCAL"


@pytest.mark.asyncio
async def test_worker_registry_selects_lowest_load_worker():
    """
    Verifies Capability-Based Selection:
    Registry filters by required capabilities and picks the healthy worker with lowest load.
    """
    registry = WorkerRegistry()

    w1 = WorkerProfile(worker_id="gpu-01", capabilities={"rag", "gpu"}, load=80.0)
    w2 = WorkerProfile(worker_id="gpu-02", capabilities={"rag", "gpu"}, load=20.0)
    w3 = WorkerProfile(worker_id="cpu-01", capabilities={"rag", "cpu"}, load=5.0)

    await registry.register_worker(w1)
    await registry.register_worker(w2)
    await registry.register_worker(w3)

    # Must select gpu-02 because it has 'gpu' capability and lower load than gpu-01
    selected = await registry.select_worker(["rag", "gpu"])
    assert selected is not None
    assert selected.worker_id == "gpu-02"