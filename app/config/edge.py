# Path: app/config/edge.py
import os
import logging

logger = logging.getLogger(__name__)

class EdgeConfig:
    """
    Phase 6E: Edge Mode Configuration.
    
    When NEUROMESH_MODE=edge is set, this configuration forces the application
    to use local SQLite instead of requiring a heavy external PostgreSQL cluster.
    """
    
    @classmethod
    def apply_edge_overrides(cls) -> None:
        if os.getenv("NEUROMESH_MODE", "").strip().lower() == "edge":
            logger.info("Edge Mode detected! Applying resource-constrained overrides.")
            
            # Override database to local SQLite
            sqlite_path = os.getenv("EDGE_SQLITE_PATH", "sqlite:///./edge_swarm.db")
            os.environ["DATABASE_URL"] = sqlite_path
            
            # Reduce default timeouts for faster failure on spotty edge networks
            os.environ["SWARM_DEFAULT_TIMEOUT"] = "10.0"
            
            # Disable heavy local embedding models to save RAM (force API usage)
            os.environ["USE_LOCAL_EMBEDDINGS"] = "false"
