# app/api/chat_stream.py
"""
Phase 5: SSE Streaming Endpoint — Streams LLM tokens back to the client
in real-time using Server-Sent Events, followed by metadata (citations,
execution steps, reflection scores).
"""

import json
import logging
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree
from pydantic import BaseModel, Field

from app.context.optimizer import ContextOptimizer
from app.llm.client import DEFAULT_MODEL, get_llm_client
from app.planner.analyzer import TaskAnalyzer
from app.reasoning.citations import CitationGenerator
from app.reasoning.reflection import ReflectionEngine
from app.retrieval.search import hybrid_search

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Streaming"])

# Module-level singletons
_optimizer = ContextOptimizer()
_reflection = ReflectionEngine()
_citation_gen = CitationGenerator()


class StreamChatRequest(BaseModel):
    """Request body for the streaming chat endpoint."""
    message: str = Field(..., min_length=1, description="User message")
    session_id: str | None = Field(default=None, description="Optional session ID")


async def _build_context(user_query: str) -> tuple[str, list[dict[str, Any]]]:
    """
    Runs the planner to decide if retrieval is needed, then fetches and
    optimizes context. Returns (optimized_context_str, raw_chunks).
    """
    analyzer = TaskAnalyzer()
    plan = await analyzer.analyze_query(user_query)

    raw_chunks: list[dict[str, Any]] = []
    tool_results: dict[str, Any] = {}

    if plan.requires_retrieval:
        try:
            raw_chunks = hybrid_search(query=user_query, top_k=5)
        except Exception as e:  # noqa: BLE001
            logger.warning("Retrieval failed during streaming: %s", e)

    # Build the retrieved context string for the optimizer
    retrieved_context = ""
    if raw_chunks:
        parts = []
        for i, chunk in enumerate(raw_chunks):
            parts.append(f"[Source {i + 1}]: {chunk.get('text', '')}")
        retrieved_context = "\n\n".join(parts)

    optimized = _optimizer.optimize(retrieved_context, tool_results)
    return optimized, raw_chunks


async def _stream_tokens(user_query: str, context: str):
    """
    Generator that yields SSE-formatted token events from the LLM.
    Collects the full response text for post-processing.
    """
    system_prompt = (
        "You are the NeuroMesh Synthesizer AI.\n"
        "Answer the user's query using ONLY the provided context.\n"
        "If you don't know the answer based on the context, say so.\n"
    )
    prompt = f"USER QUERY: {user_query}\n\n{context}"

    llm = await get_llm_client()
    stream = await llm.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        stream=True,
    )

    full_response = []

    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            full_response.append(token)
            # SSE format: "data: <json>\n\n"
            event = json.dumps({"type": "token", "content": token})
            yield f"data: {event}\n\n"

    # Return the assembled full text via a special attribute
    yield "".join(full_response)


@router.post("/chat/stream")
async def chat_stream(request: StreamChatRequest):
    """
    Streams LLM tokens via Server-Sent Events.

    Event types:
    - `token`: A single streamed token fragment
    - `metadata`: Final event with citations, reflection scores, and execution info
    - `[DONE]`: End-of-stream marker
    """

    async def event_generator():
        try:
            # 1. Build context (planner + retrieval + optimization)
            context, raw_chunks = await _build_context(request.message)

            # 2. Stream tokens from LLM
            full_response_parts = []
            async for event_or_text in _stream_tokens(request.message, context):
                if event_or_text.startswith("data: "):
                    full_response_parts_check = json.loads(event_or_text[6:].strip())
                    if full_response_parts_check.get("type") == "token":
                        full_response_parts.append(full_response_parts_check["content"])
                    yield event_or_text
                else:
                    # This is the final assembled text from the generator
                    full_text = event_or_text

            # Use the collected parts if full_text wasn't captured
            if not full_response_parts:
                full_text = ""
            else:
                full_text = "".join(full_response_parts)

            # 3. Reflection pass
            reflection_result = await _reflection.verify(
                answer=full_text,
                context=context,
                user_query=request.message,
            )

            # 4. Citation generation
            citations = _citation_gen.generate(
                answer=full_text,
                retrieved_chunks=raw_chunks,
            )

            # 5. Emit metadata event
            metadata_event = json.dumps({
                "type": "metadata",
                "reflection": reflection_result,
                "citations": citations,
                "session_id": request.session_id,
            })
            yield f"data: {metadata_event}\n\n"

            # 6. End-of-stream marker
            yield "data: [DONE]\n\n"

        except Exception as e:  # noqa: BLE001
            logger.error("Streaming chat error: %s", e)
            error_event = json.dumps({"type": "error", "message": str(e)})
            yield f"data: {error_event}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
