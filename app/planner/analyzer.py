import json
from app.llm.client import llm_client, DEFAULT_MODEL
from app.planner.schemas import ExecutionPlan

# Dynamically generate the JSON schema from Pydantic
JSON_SCHEMA = json.dumps(ExecutionPlan.model_json_schema(), indent=2)

SYSTEM_PROMPT = f"""
You are the Task Analyzer Engine for NeuroMesh, an AI orchestration system.
Your job is to read user prompts and create a structured JSON Execution Plan.

Available Tools:
1. 'rag_search': Use when the query asks about internal documents, architecture, vector data, or stored knowledge base records.
2. 'sql_query': Use when the user asks for database statistics, session logs, or relational tables.
3. 'lora_adapter': Use when specialized domain knowledge is required (e.g., 'coding', 'finance', 'legal').
4. 'direct_llm': Use for general conversation, greetings, simple questions, or basic reasoning.

STRICT JSON OUTPUT REQUIREMENT:
You MUST format your output as a JSON object strictly adhering to this schema:

{JSON_SCHEMA}

RULES:
- Do NOT rename fields or introduce keys that are not defined in the schema.
- 'requires_retrieval' must be true if 'rag_search' is used in any step.
- 'suggested_lora' must be 'coding', 'finance', 'legal', or null.
- Every step in 'steps' MUST include 'step_number', 'tool', 'description', and 'query_input'.
"""

async def analyze_task(user_query: str) -> ExecutionPlan:
    """
    Analyzes a user query using Groq LLM and returns a validated ExecutionPlan.
    """
    response = await llm_client.chat.completions.create(
        model=DEFAULT_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user", 
                "content": f"User Query: '{user_query}'\nGenerate the JSON ExecutionPlan strictly adhering to the schema."
            }
        ],
        temperature=0.0  # Set temperature to 0 for maximum deterministic output
    )

    raw_json = response.choices[0].message.content
    plan_dict = json.loads(raw_json)
    
    # Validate against Pydantic model and return
    return ExecutionPlan(**plan_dict)