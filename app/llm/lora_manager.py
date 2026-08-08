import logging
import torch
from typing import Optional, Dict, Any
from app.llm.exceptions import AdapterNotFoundError, AdapterLoadError

logger = logging.getLogger(__name__)

class LoRAManager:
    def __init__(self, base_model: Optional[Any] = None):
        """
        Initializes the manager. In production, base_model is the loaded HF Transformers model.
        """
        self.base_model = base_model
        self.active_adapter_id: Optional[str] = None
        
        # Registry mapping adapter_ids to local paths or Hugging Face Hub IDs
        self.adapter_registry: Dict[str, str] = {
            "finance": "neuromesh/finance-lora-v1",
            "legal": "neuromesh/legal-lora-v1",
            "coding": "neuromesh/coding-lora-v1",
            "research": "neuromesh/research-lora-v1"
        }

    async def load_adapter(self, adapter_id: str) -> bool:
        """
        Lazy loads a PEFT adapter. Explicitly unloads the previous one to save VRAM.
        """
        if adapter_id not in self.adapter_registry:
            logger.error(f"Adapter ID '{adapter_id}' is unknown.")
            raise AdapterNotFoundError(f"Adapter '{adapter_id}' not found in registry.")

        if self.active_adapter_id == adapter_id:
            logger.info(f"Adapter '{adapter_id}' is already loaded. Reusing from cache.")
            return True

        logger.info(f"Initializing dynamic weight swap for: '{adapter_id}'...")

        try:
            if self.base_model is not None:
                # VRAM Optimization: Unload existing adapter before loading a new one
                if self.active_adapter_id:
                    logger.info(f"Unloading active adapter '{self.active_adapter_id}' to free GPU VRAM...")
                    self.base_model.delete_adapter(self.active_adapter_id)
                    
                    # Force garbage collection on the GPU (Crucial for RTX 3050 constraints)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # Load and activate the new adapter via PEFT
                adapter_path = self.adapter_registry[adapter_id]
                self.base_model.load_adapter(adapter_path, adapter_name=adapter_id)
                self.base_model.set_adapter(adapter_id)
                
            self.active_adapter_id = adapter_id
            logger.info(f"Successfully loaded and activated adapter: '{adapter_id}'")
            return True

        except Exception as e:
            logger.error(f"Hardware/PEFT failure while loading '{adapter_id}': {str(e)}")
            raise AdapterLoadError(f"Failed to load adapter '{adapter_id}': {str(e)}")

    async def unload_all(self) -> bool:
        """Explicit manual cleanup for complete VRAM recovery."""
        if self.active_adapter_id and self.base_model is not None:
            try:
                self.base_model.delete_adapter(self.active_adapter_id)
                self.active_adapter_id = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("Successfully flushed all adapters from GPU memory.")
                return True
            except Exception as e:
                logger.error(f"Failed to cleanly unload adapters: {e}")
                return False
        return True