class AdapterNotFoundError(Exception):
    """Raised when the requested adapter_id is not found in the registry."""
    pass

class AdapterLoadError(Exception):
    """Raised when PEFT/PyTorch fails to load or inject the adapter weights."""
    pass