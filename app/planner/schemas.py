from pydantic import BaseModel, Field
from typing import List, Optional, Literal

# Allowed tools that the Orchestration Engine supports
ToolType = Literal["rag_search", "sql_query", "lora_adapter", "direct_llm"]

class TaskStep(BaseModel):
    step_number: int = Field(description="Order of execution, starting at 1")
    tool: ToolType = Field(description="The tool or execution path required for this step")
    description: str = Field(description="Brief explanation of what this step accomplishes")
    query_input: str = Field(description="The processed input string passed to the selected tool")

class ExecutionPlan(BaseModel):
    user_intent: str = Field(description="High-level summary of what the user is trying to achieve")
    requires_retrieval: bool = Field(description="True if context from the vector database is required")
    suggested_lora: Optional[str] = Field(
        default=None, 
        description="Optional domain LoRA adapter (e.g., 'coding', 'finance', 'legal', or None)"
    )
    steps: List[TaskStep] = Field(description="Ordered list of sequential steps to resolve the query")