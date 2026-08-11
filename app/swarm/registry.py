# Path: app/swarm/registry.py
import asyncio
import logging
import time
from enum import Enum

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class WorkerStatus(str, Enum):
    ONLINE = "online"
    BUSY = "busy"
    OFFLINE = "offline"
    UNHEALTHY = "unhealthy"


class HardwareProfile(BaseModel):
    device: str = Field(default="cpu", description="'cuda' or 'cpu'")
    gpu_memory_mb: int = Field(default=0)
    cpu_cores: int = Field(default=1)
    ram_mb: int = Field(default=4096)


class WorkerProfile(BaseModel):
    worker_id: str
    capabilities: set[str] = Field(default_factory=set)
    status: WorkerStatus = WorkerStatus.ONLINE
    load: float = Field(default=0.0, ge=0.0, le=100.0)
    hardware: HardwareProfile = Field(default_factory=HardwareProfile)
    last_heartbeat: float = Field(default_factory=time.time)


class WorkerRegistry:
    def __init__(self, heartbeat_timeout_seconds: float = 45.0):
        self.heartbeat_timeout_seconds = heartbeat_timeout_seconds
        self._workers: dict[str, WorkerProfile] = {}
        self._lock = asyncio.Lock()

    async def register_worker(self, profile: WorkerProfile) -> bool:
        async with self._lock:
            profile.last_heartbeat = time.time()
            profile.status = WorkerStatus.ONLINE
            self._workers[profile.worker_id] = profile
            logger.info("Registered swarm worker: %s | Capabilities: %s", profile.worker_id, profile.capabilities)
            return True

    async def unregister_worker(self, worker_id: str) -> bool:
        async with self._lock:
            if worker_id in self._workers:
                del self._workers[worker_id]
                logger.info("Unregistered swarm worker: %s", worker_id)
                return True
            return False

    async def update_heartbeat(
        self, worker_id: str, load: float, status: WorkerStatus = WorkerStatus.ONLINE
    ) -> bool:
        async with self._lock:
            worker = self._workers.get(worker_id)
            if not worker:
                return False
            worker.last_heartbeat = time.time()
            worker.load = max(0.0, min(100.0, load))
            worker.status = status
            return True

    async def select_worker(self, required_capabilities: list[str]) -> WorkerProfile | None:
        async with self._lock:
            now = time.time()
            eligible_workers: list[WorkerProfile] = []
            required_set = set(required_capabilities)

            for worker in self._workers.values():
                if now - worker.last_heartbeat > self.heartbeat_timeout_seconds:
                    worker.status = WorkerStatus.UNHEALTHY
                    continue
                if worker.status in (WorkerStatus.OFFLINE, WorkerStatus.UNHEALTHY):
                    continue
                if required_set.issubset(worker.capabilities):
                    eligible_workers.append(worker)

            if not eligible_workers:
                return None
            return min(eligible_workers, key=lambda w: w.load)

    async def get_all_workers(self) -> list[WorkerProfile]:
        async with self._lock:
            return list(self._workers.values())
