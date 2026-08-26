# app/reasoning/reflection.py
"""
Phase 5: Reflection Engine — A secondary AI verification pass that checks
generated answers against retrieved context to detect hallucinations.
"""

import logging
from typing import Any

from langsmith import traceable

from app.llm.client import DEFAULT_MODEL, get_llm_client

logger = logging.getLogger(__name__)

REFLECTION_SYSTEM_PROMPT = """\
You are a strict Verification AI for the NeuroMesh platform.
Your ONLY job is to compare a generated answer against the provided source context
and identify any claims that are NOT supported by the context.

You MUST output valid JSON with this exact schema:
{
  "confidence_score": <float between 0.0 and 1.0>,
  "is_faithful": <true if all claims are supported, false otherwise>,
  "flagged_claims": [
    {
      "claim": "<the unsupported statement>",
      "reason": "<why it is unsupported>"
    }
  ],
  "summary": "<one-sentence assessment>"
}

Rules:
- A score of 1.0 means perfect faithfulness.
- A score below 0.5 means significant hallucination detected.
- If the context is empty or the answer says "I don't know", give a score of 1.0.
- Output ONLY valid JSON. No extra text.
"""


class ReflectionEngine:
    """
    Verifies generated answers against retrieved context to prevent hallucinations.
    Uses a fast LLM pass with structured JSON output.
    """

    @traceable(name="Reflection Verification")
    async def verify(
        self,
        answer: str,
        context: str,
        user_query: str,
    ) -> dict[str, Any]:
        """
        Runs a reflection pass on the generated answer.

        Args:
            answer: The LLM-generated response to verify.
            context: The retrieved context that was used to generate the answer.
            user_query: The original user question.

        Returns:
            Dict with confidence_score, is_faithful, flagged_claims, and summary.
        """
        if not answer or not answer.strip():
            return {
                "confidence_score": 0.0,
                "is_faithful": False,
                "flagged_claims": [],
                "summary": "No answer provided to verify.",
            }

        if not context or not context.strip():
            return {
                "confidence_score": 1.0,
                "is_faithful": True,
                "flagged_claims": [],
                "summary": "No context available; answer accepted as-is.",
            }

        prompt = (
            f"USER QUERY: {user_query}\n\n"
            f"GENERATED ANSWER:\n{answer}\n\n"
            f"SOURCE CONTEXT:\n{context}"
        )

        try:
            llm = await get_llm_client()
            response = await llm.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": REFLECTION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )

            import json
            result = json.loads(response.choices[0].message.content)

            # Ensure expected keys exist
            return {
                "confidence_score": float(result.get("confidence_score", 0.5)),
                "is_faithful": bool(result.get("is_faithful", True)),
                "flagged_claims": result.get("flagged_claims", []),
                "summary": result.get("summary", "Verification completed."),
            }

        except (ValueError, KeyError, TypeError) as e:
            logger.error("Reflection engine parsing error: %s", e)
            return {
                "confidence_score": 0.5,
                "is_faithful": True,
                "flagged_claims": [],
                "summary": f"Reflection pass encountered a parsing error: {e}",
            }
        except Exception as e:  # noqa: BLE001
            logger.error("Reflection engine failed: %s", e)
            return {
                "confidence_score": 0.5,
                "is_faithful": True,
                "flagged_claims": [],
                "summary": f"Reflection pass unavailable: {e}",
            }
