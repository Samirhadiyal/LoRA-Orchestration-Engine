# app/evaluation/rag_evaluator.py
"""
Phase 7: RAG Evaluation Framework — Measures the quality of retrieval-augmented
generation using Faithfulness, Answer Relevancy, and Context Precision metrics.

Uses lightweight LLM-as-judge scoring via the same Groq endpoint to avoid
additional dependencies like deepeval or ragas.
"""

import json
import logging
from typing import Any

from langsmith import traceable

from app.llm.client import DEFAULT_MODEL, get_llm_client

logger = logging.getLogger(__name__)

EVALUATION_SYSTEM_PROMPT = """\
You are a RAG Quality Evaluator. You will be given a question, an answer, and the context 
that was used to generate the answer.

Score the answer on three dimensions. Each score must be between 0.0 and 1.0.

1. **faithfulness**: Does the answer contain ONLY information present in the context? 
   1.0 = perfectly grounded, 0.0 = pure hallucination.

2. **answer_relevancy**: Does the answer directly address the question? 
   1.0 = perfectly relevant, 0.0 = completely off-topic.

3. **context_precision**: Is the retrieved context relevant to the question? 
   1.0 = all context chunks are highly relevant, 0.0 = none are relevant.

Output ONLY valid JSON:
{
  "faithfulness": <float>,
  "answer_relevancy": <float>,
  "context_precision": <float>,
  "overall_score": <float, average of the three>,
  "reasoning": "<brief explanation of scoring>"
}
"""


class RAGEvaluator:
    """
    Evaluates RAG pipeline quality using LLM-as-judge scoring.
    No external evaluation library required — uses the project's own LLM.
    """

    @traceable(name="RAG Evaluation")
    async def evaluate(
        self,
        question: str,
        answer: str,
        context: str,
    ) -> dict[str, Any]:
        """
        Evaluates a single RAG response.

        Args:
            question: The original user question.
            answer: The generated answer.
            context: The retrieved context that was used.

        Returns:
            Dict with faithfulness, answer_relevancy, context_precision,
            overall_score, and reasoning.
        """
        if not question or not answer:
            return {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "overall_score": 0.0,
                "reasoning": "Missing question or answer.",
            }

        prompt = (
            f"QUESTION: {question}\n\n"
            f"ANSWER: {answer}\n\n"
            f"CONTEXT:\n{context or 'No context provided.'}"
        )

        try:
            llm = await get_llm_client()
            response = await llm.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": EVALUATION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )

            result = json.loads(response.choices[0].message.content)

            # Normalize and validate scores
            faithfulness = max(0.0, min(1.0, float(result.get("faithfulness", 0.5))))
            relevancy = max(0.0, min(1.0, float(result.get("answer_relevancy", 0.5))))
            precision = max(0.0, min(1.0, float(result.get("context_precision", 0.5))))
            overall = round((faithfulness + relevancy + precision) / 3, 4)

            return {
                "faithfulness": round(faithfulness, 4),
                "answer_relevancy": round(relevancy, 4),
                "context_precision": round(precision, 4),
                "overall_score": overall,
                "reasoning": result.get("reasoning", "Evaluation completed."),
            }

        except (ValueError, KeyError, TypeError) as e:
            logger.error("RAG evaluation parsing error: %s", e)
            return {
                "faithfulness": 0.5,
                "answer_relevancy": 0.5,
                "context_precision": 0.5,
                "overall_score": 0.5,
                "reasoning": f"Evaluation encountered a parsing error: {e}",
            }
        except Exception as e:  # noqa: BLE001
            logger.error("RAG evaluation failed: %s", e)
            return {
                "faithfulness": 0.5,
                "answer_relevancy": 0.5,
                "context_precision": 0.5,
                "overall_score": 0.5,
                "reasoning": f"Evaluation unavailable: {e}",
            }


    async def evaluate_batch(
        self,
        test_cases: list[dict[str, str]],
    ) -> dict[str, Any]:
        """
        Evaluates a batch of RAG test cases and returns aggregate scores.

        Args:
            test_cases: List of dicts, each with 'question', 'answer', and 'context'.

        Returns:
            Dict with individual results and aggregate averages.
        """
        results = []
        for case in test_cases:
            result = await self.evaluate(
                question=case.get("question", ""),
                answer=case.get("answer", ""),
                context=case.get("context", ""),
            )
            results.append({**case, "scores": result})

        # Compute aggregate averages
        if results:
            avg_faithfulness = sum(r["scores"]["faithfulness"] for r in results) / len(results)
            avg_relevancy = sum(r["scores"]["answer_relevancy"] for r in results) / len(results)
            avg_precision = sum(r["scores"]["context_precision"] for r in results) / len(results)
            avg_overall = sum(r["scores"]["overall_score"] for r in results) / len(results)
        else:
            avg_faithfulness = avg_relevancy = avg_precision = avg_overall = 0.0

        return {
            "total_cases": len(results),
            "aggregate_scores": {
                "faithfulness": round(avg_faithfulness, 4),
                "answer_relevancy": round(avg_relevancy, 4),
                "context_precision": round(avg_precision, 4),
                "overall_score": round(avg_overall, 4),
            },
            "individual_results": results,
        }
