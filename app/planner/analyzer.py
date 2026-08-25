from app.planner.schemas import ExecutionPlan, TaskStep


class TaskAnalyzer:
    async def analyze_query(self, query: str) -> ExecutionPlan:
        """
        Analyzes the user query and generates a structured execution plan.
        """
        # Returning a valid plan so the router can execute it!
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
                    tool="swarm_delegate",
                    description="Delegate SQL log analysis to swarm worker",
                    query_input=query
                )
            ]
        )
