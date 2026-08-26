# Path: app/swarm/workers/rag_worker.py
import logging
import time
from typing import Any

from app.swarm.registry import WorkerProfile, WorkerRegistry, WorkerStatus
from app.swarm.worker import BaseSwarmWorker, SwarmMessageBusProtocol

logger = logging.getLogger(__name__)


class RAGWorker(BaseSwarmWorker):
    """
    Phase 6C Track B: Specialized RAG Swarm Worker.

    Executes hybrid retrieval (Qdrant dense vector + BM25 keyword search) for tasks
    delegated by SwarmHandler. Operates asynchronously in a background lifecycle loop.
    """

    def __init__(
        self,
        worker_id: str,
        registry: WorkerRegistry,
        message_bus: SwarmMessageBusProtocol,
        retrieval_engine: Any = None,
        heartbeat_interval_seconds: float = 15.0,
    ):
        profile = WorkerProfile(
            worker_id=worker_id,
            capabilities={"rag", "retrieval", "embed"},
            status=WorkerStatus.ONLINE,
            load=5.0,
        )
        super().__init__(
            profile=profile,
            registry=registry,
            message_bus=message_bus,
            heartbeat_interval_seconds=heartbeat_interval_seconds,
        )
        self.retrieval_engine = retrieval_engine

    def get_current_load(self) -> float:
        """
        Returns estimated CPU/RAM load percentage for RAG indexing/embedding.
        In production, this can inspect psutil or active task queue depth.
        """
        return self.profile.load

    async def _execute_specialized_task(self, task_payload: dict, config: dict) -> dict:
        """
        Executes document retrieval for a delegated SwarmTask payload.

        Expected input schema:
            {
                "task_id": "...",
                "step_id": "...",
                "action": "rag_search",
                "metadata": {"query": "...", "top_k": 10, ...}
            }

        Returns structured result dictionary compatible with SwarmHandler.
        """
        task_id = task_payload.get("task_id", "unknown")
        step_id = task_payload.get("step_id", "unknown")
        metadata: dict[str, Any] = task_payload.get("metadata", {}) or {}
        query = metadata.get("query", "")
        top_k = int(metadata.get("top_k", 5))

        logger.info(
            "RAGWorker '%s' processing task %s (step=%s | top_k=%s)",
            self.profile.worker_id,
            task_id,
            step_id,
            top_k,
        )

        start_time = time.time()
        self.profile.load = min(100.0, self.profile.load + 25.0)

        try:
            # 1. Execute retrieval via injected RAG engine or fallback shim
            chunks = await self._execute_retrieval(query=query, top_k=top_k, metadata=metadata)

            execution_time = round(time.time() - start_time, 4)
            logger.info(
                "RAGWorker '%s' retrieved %s chunks in %ss (task=%s)",
                self.profile.worker_id,
                len(chunks),
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
                    "chunks": chunks,
                    "retrieved_count": len(chunks),
                    "execution_time_seconds": execution_time,
                    "worker_id": self.profile.worker_id,
                },
                "error": None,
            }

        except (RuntimeError, ValueError, OSError, KeyError, TypeError) as exc:
            logger.error(
                "RAGWorker '%s' failed on task %s: %s",
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
            # Cool down worker load estimation after task completion
            self.profile.load = max(5.0, self.profile.load - 25.0)

    async def _execute_retrieval(
        self, query: str, top_k: int, metadata: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """
        Internal hook interfacing with Phase 4/5 Qdrant and BM25 search engines.
        Falls back safely to clean mock chunks if engine is uninitialized in dev/tests.
        """
        if self.retrieval_engine is not None and hasattr(self.retrieval_engine, "search"):
            kwargs = {k: v for k, v in metadata.items() if k not in ("query", "top_k")}
            results = await self.retrieval_engine.search(query=query, top_k=top_k, **kwargs)
            return list(results)

        # Dev/Test Fallback: Return structured representative chunks
        return [
            {
                "id": f"chunk-{i+1}",
                "content": f"Retrieved context for query '{query}' (chunk {i+1} of {top_k}).",
                "score": round(0.95 - (i * 0.05), 2),
                "source": "neuromesh_knowledge",
            }
            for i in range(min(top_k, 3))
        ]