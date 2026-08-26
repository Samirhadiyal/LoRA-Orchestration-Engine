# app/api/evaluation.py
"""
Phase 7: Evaluation API — Exposes endpoints to evaluate RAG pipeline quality
by scoring faithfulness, answer relevancy, and context precision.
"""

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.evaluation.rag_evaluator import RAGEvaluator

router = APIRouter(prefix="/evaluate", tags=["Evaluation"])

_evaluator = RAGEvaluator()


class EvaluateRequest(BaseModel):
    """Request body for single RAG evaluation."""
    question: str = Field(..., description="The original user question")
    answer: str = Field(..., description="The generated answer to evaluate")
    context: str = Field(default="", description="The retrieved context used for generation")


class BatchEvaluateRequest(BaseModel):
    """Request body for batch RAG evaluation."""
    test_cases: list[EvaluateRequest] = Field(
        ..., description="List of question/answer/context triples to evaluate"
    )


@router.post("/")
async def evaluate_single(body: EvaluateRequest) -> dict[str, Any]:
    """
    Evaluates a single RAG response for faithfulness, relevancy, and precision.
    """
    return await _evaluator.evaluate(
        question=body.question,
        answer=body.answer,
        context=body.context,
    )


@router.post("/batch")
async def evaluate_batch(body: BatchEvaluateRequest) -> dict[str, Any]:
    """
    Evaluates a batch of RAG responses and returns aggregate quality scores.
    """
    test_cases = [
        {"question": tc.question, "answer": tc.answer, "context": tc.context}
        for tc in body.test_cases
    ]
    return await _evaluator.evaluate_batch(test_cases)
