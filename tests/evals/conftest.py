# Path: tests/evals/conftest.py
import pytest
from datasets import Dataset
from typing import Dict, Any, List


@pytest.fixture(scope="session")
def rag_benchmark_dataset() -> Dataset:
    """
    Provides a standardized evaluation dataset for RAGAS metrics testing.
    Mirrors expected chunks from Qdrant 'neuromesh_knowledge' (384-dim Cosine).
    """
    sample_data = {
        "question": [
            "How does NeuroMesh manage LoRA adapter VRAM usage on mobile GPUs?",
            "What is the priority order in the Context Optimizer?",
        ],
        "contexts": [
            [
                "LoRAManager explicitly unloads active adapters using delete_adapter and invokes torch.cuda.empty_cache() before loading new weights to respect RTX 3050 Laptop VRAM limits.",
                "PEFT models are loaded with FP16 or 4-bit NF4 quantization to conserve GPU memory.",
            ],
            [
                "The Context Optimizer prioritizes execution tool_results over retrieved static knowledge chunks.",
                "It enforces a hard character budget of 12,000 chars and removes duplicate chunks using MD5 or semantic similarity.",
            ],
        ],
        "answer": [
            "NeuroMesh manages VRAM on mobile GPUs by unloading active adapters with delete_adapter, calling torch.cuda.empty_cache(), and utilizing 4-bit NF4 quantization for PEFT weights.",
            "The Context Optimizer prioritizes tool_results over retrieved static chunks and enforces a 12,000 character maximum budget.",
        ],
        "ground_truth": [
            "NeuroMesh unloads active adapters, calls torch.cuda.empty_cache(), and uses 4-bit quantization to fit RTX 3050 VRAM limits.",
            "Tool results take priority over static retrieved chunks within a 12,000 character budget.",
        ],
    }
    return Dataset.from_dict(sample_data)


@pytest.fixture
def mock_execution_trajectory() -> List[Dict[str, Any]]:
    """
    Mock LangGraph execution trajectory to verify StepStatus transitions
    and ensure zero infinite retry loops.
    """
    return [
        {"step_id": "step_1", "action": "rag_search", "status": "completed", "error": None},
        {"step_id": "step_2", "action": "sql_query", "status": "completed", "error": None},
        {"step_id": "step_3", "action": "lora_adapter", "status": "completed", "error": None},
        {"step_id": "step_4", "action": "direct_llm", "status": "completed", "error": None},
    ]