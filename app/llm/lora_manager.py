import asyncio
import gc
import logging
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from app.llm.exceptions import AdapterLoadError, AdapterNotFoundError

logger = logging.getLogger(__name__)


class LoRAManager:
    """
    QLoRA 4-bit LoRA manager with active-adapter caching and VRAM flushing.

    Designed for RTX 3050 Laptop GPU constraints. Only one adapter is active at
    a time, and adapter swaps force Python/CUDA cleanup before loading weights.
    """

    def __init__(
        self,
        base_model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        adapters_dir: str = "./adapters",
    ) -> None:
        self.base_model_name = base_model_name
        self.adapters_dir = Path(adapters_dir)
        self.model: PeftModel | None = None
        self._active_adapter_id: str | None = None
        self._lock = asyncio.Lock()

    async def load_adapter(self, adapter_id: str) -> bool:
        """
        Load a LoRA adapter, short-circuiting if it is already active.
        """
        async with self._lock:
            if self._active_adapter_id == adapter_id and self.model is not None:
                logger.debug("Adapter '%s' already active. Skipping reload.", adapter_id)
                return True

            try:
                await self._unload_all_internal()
                adapter_path = self.adapters_dir / adapter_id
                if not adapter_path.exists():
                    raise FileNotFoundError(adapter_path)

                logger.info("Loading QLoRA adapter: %s", adapter_id)
                await asyncio.to_thread(
                    self._load_adapter_weights, adapter_id, adapter_path
                )
                self._active_adapter_id = adapter_id
                return True

            except FileNotFoundError as exc:
                logger.error("Adapter directory not found: %s", adapter_id)
                raise AdapterNotFoundError(
                    f"Adapter '{adapter_id}' does not exist."
                ) from exc
            except RuntimeError as exc:
                logger.error("Failed to load adapter '%s': %s", adapter_id, exc)
                raise AdapterLoadError(
                    f"Error loading LoRA adapter '{adapter_id}'."
                ) from exc

    async def unload_all(self) -> bool:
        """Unload the active adapter and flush CUDA memory."""
        async with self._lock:
            return await self._unload_all_internal()

    async def _unload_all_internal(self) -> bool:
        if self._active_adapter_id is None and self.model is None:
            return True

        logger.info("Unloading active LoRA adapter and flushing GPU memory.")
        try:
            if self.model is not None:
                if hasattr(self.model, "delete_adapter") and self._active_adapter_id:
                    self.model.delete_adapter(self._active_adapter_id)
                del self.model
                self.model = None

            self._active_adapter_id = None
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            return True
        except RuntimeError as exc:
            logger.error("Error during adapter unloading: %s", exc)
            return False

    def _load_adapter_weights(self, adapter_id: str, adapter_path: Path) -> None:
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
