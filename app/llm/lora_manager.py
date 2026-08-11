# Path: app/llm/lora_manager.py
import asyncio
import gc
import logging
from typing import Optional
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from app.llm.exceptions import AdapterLoadError, AdapterNotFoundError

logger = logging.getLogger(__name__)


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
    Phase 5 Track B: QLoRA 4-Bit LoRA Manager with LRU Active Adapter Caching.
    Strictly designed for RTX 3050 Laptop VRAM constraints (4GB-6GB).
    """

    def __init__(self, base_model_name: str, adapters_dir: str = "./adapters"):
        self.base_model_name = base_model_name
        self.adapters_dir = adapters_dir
        self.model: Optional[PeftModel] = None
        self._active_adapter_id: Optional[str] = None
        self._lock = asyncio.Lock()

    async def load_adapter(self, adapter_id: str) -> bool:
        """
        Asynchronously loads a LoRA adapter. Short-circuits if already loaded (LRU Cache).
        Enforces clean VRAM garbage collection before swapping adapters.
        """
        async with self._lock:
            # 1. LRU Short-Circuit: Zero I/O if adapter is already active
            if self._active_adapter_id == adapter_id and self.model is not None:
                logger.debug(f"Adapter '{adapter_id}' already active in cache. Skipping reload.")
                return True

            try:
                # 2. Flush any existing adapter from VRAM first
                await self._unload_all_internal()

                logger.info(f"Loading QLoRA adapter: {adapter_id}")
                adapter_path = f"{self.adapters_dir}/{adapter_id}"

                # Offload blocking disk/GPU load to thread pool to prevent event loop blocking
                await asyncio.to_thread(
                    self._load_adapter_weights, adapter_id, adapter_path
                )

                self._active_adapter_id = adapter_id
                return True

            except FileNotFoundError as exc:
                logger.error(f"Adapter directory not found: {adapter_id}")
                raise AdapterNotFoundError(f"Adapter '{adapter_id}' does not exist.") from exc
            except Exception as exc:
                logger.error(f"Failed to load adapter '{adapter_id}': {str(exc)}")
                raise AdapterLoadError(f"Error loading LoRA adapter '{adapter_id}'.") from exc

    async def unload_all(self) -> bool:
        """
        Public interface: Explicitly unloads active adapters and flushes CUDA VRAM cache.
        """
        async with self._lock:
            return await self._unload_all_internal()

    async def _unload_all_internal(self) -> bool:
        """Internal unlocked unload routine to avoid deadlocks."""
        if self._active_adapter_id is None and self.model is None:
            return True

        logger.info("Unloading active LoRA adapter and flushing RTX 3050 VRAM...")
        try:
            if self.model is not None:
                # Safely detach PEFT adapter layers
                if hasattr(self.model, "delete_adapter") and self._active_adapter_id:
                    self.model.delete_adapter(self._active_adapter_id)
                del self.model
                self.model = None

            self._active_adapter_id = None

            # Aggressive VRAM reclamation for RTX 3050 mobile GPU
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            return True
        except Exception as exc:
            logger.error(f"Error during adapter unloading: {str(exc)}")
            return False

    def _load_adapter_weights(self, adapter_id: str, adapter_path: str) -> None:
        """Synchronous weight loader executed inside a thread worker."""
        # Optional: Initialize base model in 4-bit QLoRA if not already cached globally
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        self.model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            adapter_name=adapter_id,
            torch_dtype=torch.float16,
        )
        self.model.eval()
