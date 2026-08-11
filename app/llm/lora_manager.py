import gc
import logging
from collections import OrderedDict
from typing import Optional, Dict, Any

try:
    import torch
except ImportError:
    torch = None

logger = logging.getLogger("neuromesh.lora_manager")


class LoRAManager:
    """
    Manages QLoRA 4-bit / 8-bit adapter loading, LRU caching, and hot-swapping
    with explicit CUDA VRAM cleanup for memory safety.
    """

    def __init__(self, max_cache_size: int = 3, load_in_4bit: bool = True):
        self.active_adapter: Optional[str] = None
        self.max_cache_size = max_cache_size
        self.load_in_4bit = load_in_4bit
        
        # LRU Cache for adapter metadata / loaded weights
        self.adapter_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.supported_adapters = {"coding", "finance", "legal"}

    async def load_adapter(self, adapter_name: str) -> bool:
        """
        Loads or hot-swaps the requested LoRA adapter using LRU cache lookup.
        """
        if not adapter_name:
            logger.info("No LoRA adapter requested. Using base model.")
            return True

        cleaned_name = adapter_name.lower().strip()

        # 1. Check if already active
        if self.active_adapter == cleaned_name:
            logger.info(f"Adapter '{cleaned_name}' is already active.")
            return True

        # 2. Check LRU Cache
        if cleaned_name in self.adapter_cache:
            logger.info(f"LRU Cache Hit: Activating pre-loaded adapter '{cleaned_name}'")
            self.adapter_cache.move_to_end(cleaned_name)
            self.active_adapter = cleaned_name
            return True

        # 3. Cache Miss: Free VRAM if cache is full
        if len(self.adapter_cache) >= self.max_cache_size:
            lru_adapter, _ = self.adapter_cache.popitem(last=False)
            logger.info(f"LRU Cache Full. Evicting oldest adapter: '{lru_adapter}'")
            self._clear_vram()

        # 4. Load requested adapter
        try:
            logger.info(f"Loading weights for '{cleaned_name}' (4-Bit Quantized={self.load_in_4bit})...")
            
            # Simulated adapter weight payload / PEFT reference
            adapter_payload = {
                "adapter_name": cleaned_name,
                "quantization": "4bit" if self.load_in_4bit else "8bit",
                "status": "loaded"
            }

            self.adapter_cache[cleaned_name] = adapter_payload
            self.active_adapter = cleaned_name
            logger.info(f"Successfully loaded and cached adapter '{cleaned_name}'.")
            return True

        except Exception as e:
            logger.error(f"Failed to load adapter '{adapter_name}': {e}")
            return False

    def unload_all(self) -> None:
        """
        Unloads all adapters, clears LRU cache, and flushes PyTorch CUDA memory.
        """
        logger.info("Unloading all LoRA adapters and purging cache...")
        self.adapter_cache.clear()
        self.active_adapter = None
        self._clear_vram()

    def _clear_vram(self) -> None:
        """
        Safeguard: Forces garbage collection and empties CUDA cache.
        """
        gc.collect()
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA VRAM cache successfully cleared.")

    def get_active_adapter(self) -> Optional[str]:
        return self.active_adapter

    def get_cached_adapters(self) -> list:
        return list(self.adapter_cache.keys())