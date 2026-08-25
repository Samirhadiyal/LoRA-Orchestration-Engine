# Path: app/swarm/workers/mcp_worker.py
import logging
import time
from typing import Any

from app.swarm.registry import WorkerProfile, WorkerRegistry, WorkerStatus
from app.swarm.worker import BaseSwarmWorker, SwarmMessageBusProtocol

logger = logging.getLogger(__name__)


class MCPWorker(BaseSwarmWorker):
    """
    Phase 6C Track B: Specialized MCP / External Tool Swarm Worker.

    Executes external third-party API lookups and Model Context Protocol (MCP)
    tool invocations delegated by SwarmHandler in a background lifecycle loop.
    """

    def __init__(
        self,
        worker_id: str,
        registry: WorkerRegistry,
        message_bus: SwarmMessageBusProtocol,
        mcp_client: Any = None,
        heartbeat_interval_seconds: float = 15.0,
    ):
        profile = WorkerProfile(
            worker_id=worker_id,
            capabilities={"mcp", "external_api", "tools"},
            status=WorkerStatus.ONLINE,
            load=5.0,
        )
        super().__init__(
            profile=profile,
            registry=registry,
            message_bus=message_bus,
            heartbeat_interval_seconds=heartbeat_interval_seconds,
        )
        self.mcp_client = mcp_client

    def get_current_load(self) -> float:
        """Returns estimated network I/O and concurrent tool execution load."""
        return self.profile.load

    async def _execute_specialized_task(self, task_payload: dict, config: dict) -> dict:
        """
        Executes external MCP tool lookups for a delegated SwarmTask payload.

        Expected input schema:
            {
                "task_id": "...",
                "step_id": "...",
                "action": "mcp_tool",
                "metadata": {
                    "tool_name": "weather_lookup",
                    "arguments": {"location": "Rajkot, IN"}
                }
            }

        Returns structured result dictionary compatible with SwarmHandler.
        """
        task_id = task_payload.get("task_id", "unknown")
        step_id = task_payload.get("step_id", "unknown")
        metadata: dict[str, Any] = task_payload.get("metadata", {}) or {}

        tool_name = str(metadata.get("tool_name", "default_tool"))
        arguments: dict[str, Any] = metadata.get("arguments", {}) or {}

        logger.info(
            "MCPWorker '%s' executing tool '%s' for task %s (step=%s)",
            self.profile.worker_id,
            tool_name,
            task_id,
            step_id,
        )

        start_time = time.time()
        self.profile.load = min(100.0, self.profile.load + 15.0)

        try:
            # 1. Execute external tool call via injected client or fallback shim
            output = await self._execute_tool(
                tool_name=tool_name, arguments=arguments, metadata=metadata
            )

            execution_time = round(time.time() - start_time, 4)
            logger.info(
                "MCPWorker '%s' completed tool '%s' in %ss (task=%s)",
                self.profile.worker_id,
                tool_name,
                execution_time,
                task_id,
            )

            # 2. Return structured success payload
            return {
                "task_id": task_id,
                "step_id": step_id,
                "worker_id": self.profile.worker_id,
                "success": True,
                "result": {
                    "tool_name": tool_name,
                    "output": output,
                    "execution_time_seconds": execution_time,
                    "worker_id": self.profile.worker_id,
                },
                "error": None,
            }

        except (RuntimeError, ValueError, OSError, KeyError, TypeError) as exc:
            logger.error(
                "MCPWorker '%s' failed on tool '%s' (task %s): %s",
                self.profile.worker_id,
                tool_name,
                task_id,
                exc,
            )
            return {
                "task_id": task_id,
                "step_id": step_id,
                "worker_id": self.profile.worker_id,
                "success": False,
                "result": {},
                "error": str(exc),
            }

        finally:
            self.profile.load = max(5.0, self.profile.load - 15.0)

    async def _execute_tool(
        self, tool_name: str, arguments: dict[str, Any], metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Internal hook interfacing with Model Context Protocol (MCP) clients or REST endpoints.
        Falls back safely to representative tool payloads if uninitialized.
        """
        if self.mcp_client is not None and hasattr(self.mcp_client, "call_tool"):
            # Strip explicit keys from metadata so keyword unpacking does not collide
            clean_metadata = {
                k: v for k, v in metadata.items() if k not in ("tool_name", "arguments")
            }
            return await self.mcp_client.call_tool(
                tool_name=tool_name, arguments=arguments, **clean_metadata
            )

        # Dev/Test Fallback: Return representative tool output
        return {
            "status": "ok",
            "tool_executed": tool_name,
            "received_args": arguments,
            "data": f"Mock external payload generated by {tool_name}",
        }
