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

        await self.registry.unregister_worker(self.profile.worker_id)
        logger.info("Worker node shutdown cleanly: %s", self.profile.worker_id)

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

    @abstractmethod
    async def process_task(self, task_payload: dict) -> dict:
        """
        Abstract execution hook for domain specialists (GPU LoRA, RAG, SQL, MCP).
        Must receive a minimal task payload and return a structured result dictionary.
        """
        ...