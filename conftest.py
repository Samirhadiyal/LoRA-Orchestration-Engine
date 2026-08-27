import os
import sys
import types
from pathlib import Path

# 1. WINDOWS PATH FIX: Ensure project root is in sys.path before any test imports
root_dir = Path(__file__).parent.resolve()
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# 2. OPENAI CREDENTIALS FIX: Prevent Ragas from crashing when OPENAI_API_KEY is not set
os.environ.setdefault("OPENAI_API_KEY", "sk-mock-dummy-key-for-offline-testing-only-0000000000")

# 3. RAGAS COMPATIBILITY SHIM: Prevent Ragas from crashing on legacy VertexAI imports
if "langchain_community.chat_models.vertexai" not in sys.modules:
    dummy_vertexai = types.ModuleType("langchain_community.chat_models.vertexai")
    dummy_vertexai.ChatVertexAI = type("ChatVertexAI", (object,), {})  # type: ignore[attr-defined]
    sys.modules["langchain_community.chat_models.vertexai"] = dummy_vertexai
