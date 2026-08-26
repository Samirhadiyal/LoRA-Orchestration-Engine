import asyncio
import logging
import os
from app.swarm.registry import WorkerRegistry
from app.swarm.bus import RedisSwarmBus
from app.swarm.workers.rag_worker import RAGWorker
from app.swarm.workers.sql_worker import SQLWorker
from app.swarm.workers.mcp_worker import MCPWorker
from app.config.edge import EdgeConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def start_workers():
    # Apply Edge constraints (SQLite fallback)
    EdgeConfig.apply_edge_overrides()
    
    # Connect to the distributed Redis bus
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    bus = RedisSwarmBus(redis_url=redis_url)
    registry = WorkerRegistry()
    
    # Initialize domain workers
    rag = RAGWorker(worker_id="rag-edge-01", registry=registry, message_bus=bus)
    sql = SQLWorker(worker_id="sql-edge-01", registry=registry, message_bus=bus)
    mcp = MCPWorker(worker_id="mcp-edge-01", registry=registry, message_bus=bus)
    
    logger.info("Starting Edge Swarm Workers connected to %s", redis_url)
    await asyncio.gather(rag.start(), sql.start(), mcp.start())
    
    while True:
        await asyncio.sleep(3600)

if __name__ == "__main__":
    try:
        asyncio.run(start_workers())
    except KeyboardInterrupt:
        logger.info("Swarm Workers shutting down.")
