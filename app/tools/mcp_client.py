from typing import Any


class MCPClient:
    """
    Handles connections and data retrieval from external tools and APIs.
    """

    async def fetch_external_data(self, tool_name: str, query: str) -> dict[str, Any]:
        """
        Routes requests to specific external tools based on tool_name.
        """
        try:
            cleaned_tool = tool_name.lower().strip()

            if "search" in cleaned_tool or "web" in cleaned_tool:
                return await self._mock_web_search(query)
            elif "weather" in cleaned_tool:
                return await self._fetch_weather(query)
            else:
                return await self._default_external_lookup(tool_name, query)

        except (ConnectionError, TimeoutError, ValueError, RuntimeError) as e:
            return {
                "success": False,
                "tool_name": tool_name,
                "error": f"Failed to execute MCP tool '{tool_name}': {e!s}"
            }

    async def _mock_web_search(self, query: str) -> dict[str, Any]:
        """Simulates an external web search query."""
        return {
            "success": True,
            "tool": "web_search",
            "query": query,
            "results": [
                {
                    "title": f"Search result for {query}",
                    "snippet": f"Latest external data and news regarding '{query}'.",
                    "source": "https://api.external-mcp.org/search"
                }
            ]
        }

    async def _fetch_weather(self, query: str) -> dict[str, Any]:
        """Example endpoint fetching real/mock weather info."""
        return {
            "success": True,
            "tool": "weather_api",
            "location": query,
            "temperature": "24°C",
            "condition": "Partly Cloudy"
        }

    async def _default_external_lookup(self, tool_name: str, query: str) -> dict[str, Any]:
        """Fallback router for unspecified external tools."""
        return {
            "success": True,
            "tool": tool_name,
            "query": query,
            "data": f"External payload generated for query '{query}' via tool '{tool_name}'"
        }