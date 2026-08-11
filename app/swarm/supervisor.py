# Path: app/swarm/supervisor.py
import logging
import os
from typing import Any

from app.router.state import ExecutionStep

logger = logging.getLogger(__name__)


class SwarmSupervisorPolicy:
    """
    Phase 6A Track B: Swarm Supervisor Decision Policy Engine.
    
    Determines whether an ExecutionStep should execute:
    - 'LOCAL': Fast local path (simple RAG, direct generation, cached lookups, or swarm disabled).
    - 'SWARM': Delegated worker path (heavy RAG, specialized GPU LoRA, external MCP, heavy SQL).
    """

    def __init__(self, swarm_enabled_default: bool = False):
        self._default_enabled = swarm_enabled_default

    def is_swarm_enabled(self) -> bool:
        """Checks environment toggle NEUROMESH_SWARM_ENABLED (default: false)."""
        env_val = os.getenv("NEUROMESH_SWARM_ENABLED", "").strip().lower()
        if env_val in ("true", "1", "yes"):
            return True
        if env_val in ("false", "0", "no"):
            return False
        return self._default_enabled

    def decide_execution_route(self, step: ExecutionStep) -> str:
        """
        Evaluates step action and metadata to decide between LOCAL and SWARM.
        Returns strictly 'LOCAL' or 'SWARM'.
        """
        action = (step.action or "").strip().lower()
        metadata: dict[str, Any] = getattr(step, "metadata", {}) or {}
        
        if not self.is_swarm_enabled():
            logger.debug("Swarm disabled via config. Routing step '%s' to LOCAL.", step.step_id)
            return "LOCAL"

        action = (step.action or "").strip().lower()
        metadata: dict[str, Any] = getattr(step, "metadata", {}) or {}

        # 1. Specialized GPU LoRA adapter execution -> ALWAYS SWARM when enabled
        if action == "lora_adapter":
            logger.info("Routing LoRA adapter step '%s' to SWARM worker.", step.step_id)
            return "SWARM"

        # 2. External MCP tool calls (network I/O / APIs) -> ALWAYS SWARM
        if action in ("mcp_tool", "external_api", "swarm_delegate"):
            logger.info("Routing external MCP step '%s' to SWARM worker.", step.step_id)
            return "SWARM"

        # 3. RAG Retrieval -> SWARM only if heavy RAG requested (e.g. multi-doc / high top_k)
        if action == "rag_search":
            is_heavy = metadata.get("heavy_rag", False) or metadata.get("top_k", 5) > 10
            if is_heavy:
                logger.info("Routing heavy RAG step '%s' to SWARM worker.", step.step_id)
                return "SWARM"
            return "LOCAL"

        # 4. SQL Queries -> SWARM only for large aggregations or long-running analytics
        if action == "sql_query":
            is_heavy_sql = metadata.get("analytics", False) or metadata.get("heavy_query", False)
            if is_heavy_sql:
                logger.info("Routing heavy SQL step '%s' to SWARM worker.", step.step_id)
                return "SWARM"
            return "LOCAL"

        # 5. Standard direct LLM generation -> Keep LOCAL by default
        logger.debug("Routing standard step '%s' (%s) to LOCAL execution.", step.step_id, action)
        return "LOCAL"