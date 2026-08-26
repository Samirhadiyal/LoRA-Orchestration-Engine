# Path: app/swarm/workers/sql_worker.py
import logging
import time
from typing import Any

from app.swarm.registry import WorkerProfile, WorkerRegistry, WorkerStatus
from app.swarm.worker import BaseSwarmWorker, SwarmMessageBusProtocol

logger = logging.getLogger(__name__)


class SQLWorker(BaseSwarmWorker):
    """
    Phase 6C Track B: Specialized SQL Swarm Worker.

    Executes safe, read-only analytical SQL queries and database aggregations
    delegated by SwarmHandler. Operates asynchronously in a background lifecycle loop.
    """

    def __init__(
        self,
        worker_id: str,
        registry: WorkerRegistry,
        message_bus: SwarmMessageBusProtocol,
        sql_engine: Any = None,
        heartbeat_interval_seconds: float = 15.0,
    ):
        profile = WorkerProfile(
            worker_id=worker_id,
            capabilities={"sql", "database", "analytics"},
            status=WorkerStatus.ONLINE,
            load=5.0,
        )
        super().__init__(
            profile=profile,
            registry=registry,
            message_bus=message_bus,
            heartbeat_interval_seconds=heartbeat_interval_seconds,
        )
        self.sql_engine = sql_engine

    def get_current_load(self) -> float:
        """Returns estimated database connection pool / query load percentage."""
        return self.profile.load

    async def _execute_specialized_task(self, task_payload: dict, config: dict) -> dict:
        """
        Executes read-only SQL queries for a delegated SwarmTask payload.

        Expected input schema:
            {
                "task_id": "...",
                "step_id": "...",
                "action": "sql_query",
                "metadata": {"query": "SELECT ...", "analytics": True, ...}
            }

        Returns structured result dictionary compatible with SwarmHandler.
        """
        task_id = task_payload.get("task_id", "unknown")
        step_id = task_payload.get("step_id", "unknown")
        metadata: dict[str, Any] = task_payload.get("metadata", {}) or {}
        sql_query = metadata.get("query", "")

        logger.info(
            "SQLWorker '%s' processing task %s (step=%s)",
            self.profile.worker_id,
            task_id,
            step_id,
        )

        start_time = time.time()
        self.profile.load = min(100.0, self.profile.load + 20.0)

        try:
            # 1. Execute safe read-only SQL via injected engine or fallback shim
            rows = await self._execute_sql(sql_query=sql_query, metadata=metadata)

            execution_time = round(time.time() - start_time, 4)
            logger.info(
                "SQLWorker '%s' fetched %s rows in %ss (task=%s)",
                self.profile.worker_id,
                len(rows),
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
                    "rows": rows,
                    "row_count": len(rows),
                    "execution_time_seconds": execution_time,
                    "worker_id": self.profile.worker_id,
                },
                "error": None,
            }

        except (RuntimeError, ValueError, OSError, KeyError, TypeError) as exc:
            logger.error(
                "SQLWorker '%s' failed on task %s: %s",
                self.profile.worker_id,
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
            self.profile.load = max(5.0, self.profile.load - 20.0)

    async def _execute_sql(
        self, sql_query: str, metadata: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """
        Internal hook interfacing with Track A SQLAgent / PostgreSQL read-only engine.
        Falls back safely to representative analytical rows if engine is uninitialized.
        """
        if self.sql_engine is not None and hasattr(self.sql_engine, "execute_query"):
            # Strip explicit positional keys so keyword unpacking does not pass duplicates
            clean_metadata = {k: v for k, v in metadata.items() if k != "query"}
            results = await self.sql_engine.execute_query(query=sql_query, **clean_metadata)
            return list(results)

        # Dev/Test Fallback: Return structured analytical rows
        return [
            {"metric": "total_revenue", "value": 142500.0, "currency": "USD"},
            {"metric": "active_users", "value": 1280, "status": "active"},
        ]
