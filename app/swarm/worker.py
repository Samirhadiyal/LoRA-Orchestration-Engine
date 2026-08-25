# Path: app/swarm/worker.py
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Protocol

from app.swarm.registry import WorkerProfile, WorkerRegistry, WorkerStatus

logger = logging.getLogger(__name__)


# Protocol contract matching Engineer A's SwarmMessageBus (app/swarm/bus.py)
class SwarmMessageBusProtocol(Protocol):
    async def listen_for_task(self, worker_id: str, timeout: float = 5.0) -> dict | None:
        ...

    async def publish_result(self, result_payload: dict) -> bool:
        ...


class BaseSwarmWorker(ABC):
    """
    Phase 6A Track B: Abstract Swarm Worker Node.
    
    Provides the background lifecycle loop for specialized worker nodes:
    - Periodically reports heartbeats and load metrics to WorkerRegistry.
    - Polls the durable Redis Streams message bus for assigned tasks.
    - Executes domain-specific logic via abstract process_task().
    """

    def __init__(
        self,
        profile: WorkerProfile,
        registry: WorkerRegistry,
        message_bus: SwarmMessageBusProtocol,
        heartbeat_interval_seconds: float = 15.0,
    ):
        self.profile = profile
        self.registry = registry
        self.bus = message_bus
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self._is_running = False
        self._heartbeat_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Registers worker profile and starts background heartbeat loop."""
        self._is_running = True
        await self.registry.register_worker(self.profile)
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._task_poll_task = asyncio.create_task(self._task_loop())
        logger.info("Worker node online: %s | Capabilities: %s", self.profile.worker_id, self.profile.capabilities)

    async def stop(self) -> None:
        """Gracefully stops heartbeat loop and unregisters worker node."""
        self._is_running = False
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except (asyncio.CancelledError, RuntimeError):
                pass
                
        if getattr(self, "_task_poll_task", None):
            self._task_poll_task.cancel()
            try:
                await self._task_poll_task
            except (asyncio.CancelledError, RuntimeError):
                pass

        await self.registry.unregister_worker(self.profile.worker_id)
        logger.info("Worker node shutdown cleanly: %s", self.profile.worker_id)

    async def _task_loop(self) -> None:
        """Background loop polling the durable message bus for assigned tasks."""
        while self._is_running:
            try:
                task = await self.bus.listen_for_task(self.profile.worker_id, timeout=1.0)
                if task:
                    logger.info("Worker %s received task %s", self.profile.worker_id, task["task_id"])
                    try:
                        result_dict = await self.process_task(task)
                        success = True
                        error = None
                    except (asyncio.TimeoutError, ConnectionError, RuntimeError) as e:
                        logger.exception("Task processing failed")
                        result_dict = {}
                        success = False
                        error = str(e)
                    
                    await self.bus.publish_result({
                        "task_id": task["task_id"],
                        "step_id": task["step_id"],
                        "worker_id": self.profile.worker_id,
                        "success": success,
                        "result": result_dict,
                        "error": error
                    })
            except (asyncio.TimeoutError, ConnectionError, RuntimeError) as e:
                logger.error("Error in task loop: %s", e)
                await asyncio.sleep(1.0)

    async def _heartbeat_loop(self) -> None:
        """Background loop dispatching telemetry heartbeats to the registry."""
        while self._is_running:
            try:
                current_load = self.get_current_load()
                await self.registry.update_heartbeat(
                    worker_id=self.profile.worker_id,
                    load=current_load,
                    status=WorkerStatus.ONLINE if current_load < 90.0 else WorkerStatus.BUSY,
                )
            except (RuntimeError, ConnectionError, OSError) as exc:
                logger.error("Heartbeat update failed for worker %s: %s", self.profile.worker_id, exc)
            
            await asyncio.sleep(self.heartbeat_interval_seconds)

    def get_current_load(self) -> float:
        """
        Returns estimated CPU/GPU load percentage.
        Can be overridden by hardware-specific subclasses.
        """
        return 10.0

    async def process_task(self, task_payload: dict) -> dict:
        """
        Execution hook for domain specialists (GPU LoRA, RAG, SQL, MCP).
        Extracts run_id from the payload and injects it into LangChain callback config.
        """
        run_id = task_payload.get("run_id")
        config = {"run_id": run_id} if run_id else {}
        
        return await self._execute_specialized_task(task_payload, config)

    @abstractmethod
    async def _execute_specialized_task(self, task_payload: dict, config: dict) -> dict:
        """
        Abstract execution hook for domain specialists.
        Must receive a minimal task payload and return a structured result dictionary.
        """
        ...