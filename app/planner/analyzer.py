import json
import logging
from langsmith import traceable
from pydantic import ValidationError

from app.planner.schemas import ExecutionPlan, TaskStep
from app.llm.client import get_llm_client, DEFAULT_MODEL

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a master planner for the NeuroMesh Orchestration Engine.
Your job is to analyze the user's query and output a structured execution plan in JSON format.

The JSON MUST conform to this exact schema (omitting comments):
{
  "user_intent": "High-level summary of what the user is trying to achieve",
  "requires_retrieval": true/false, // True if context from knowledge base is needed
  "suggested_lora": null, // Optional domain LoRA adapter ('coding', 'finance', 'legal', or null)
  "steps": [
    {
      "step_number": 1,
      "tool": "rag_search", // Allowed tools: rag_search, sql_query, lora_adapter, direct_llm, swarm_delegate
      "description": "Brief explanation",
      "query_input": "Processed string to pass to the tool",
      "worker_capability": null // Only use if tool is swarm_delegate (e.g. 'sql' or 'mcp')
    }
  ]
}

Available tools for steps:
- rag_search: For general knowledge or vector lookups.
- swarm_delegate: For complex execution. Must set worker_capability to 'sql' for database/log analysis, or 'mcp' for external API calls and tools.
- direct_llm: For basic conversational replies or final synthesis.

If the user asks for explanations, definitions, technical concepts, or general knowledge, you MUST generate a `rag_search` action as the first step to retrieve context. Never attempt to answer technical questions from memory. Always follow a `rag_search` or `swarm_delegate` step with a final `direct_llm` step for synthesis.
Output ONLY valid JSON.
"""

class TaskAnalyzer:
    @traceable(name="Dynamic Planner")
    async def analyze_query(self, query: str) -> ExecutionPlan:
        """
        Analyzes the user query and generates a structured execution plan dynamically.
        """
        try:
            llm = await get_llm_client()
            response = await llm.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": query}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            
            raw_json = response.choices[0].message.content
            plan = ExecutionPlan.model_validate_json(raw_json)
            
            # Ensure there's a final synthesis step if not present
            if not any(step.tool == "direct_llm" for step in plan.steps):
                plan.steps.append(
                    TaskStep(
                        step_number=len(plan.steps) + 1,
                        tool="direct_llm",
                        description="Synthesize final response for the user",
                        query_input=query,
                        worker_capability=None
                    )
                )
                
            return plan
        except (ValidationError, ValueError, KeyError) as e:
            logger.error(f"Planner failed to generate valid JSON plan: {e}. Falling back to default plan.")
            return ExecutionPlan(
                user_intent=query,
                requires_retrieval=True,
                suggested_lora=None,
                steps=[
                    TaskStep(
                        step_number=1,
                        tool="rag_search",
                        description="Retrieve context from knowledge base",
                        query_input=query
                    ),
                    TaskStep(
                        step_number=2,
                        tool="direct_llm",
                        description="Synthesize a generic fallback response",
                        query_input=query
                    )
                ]
            )
