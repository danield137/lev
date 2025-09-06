from __future__ import annotations

import csv
import datetime
import io
import json
import logging
from dataclasses import dataclass
from typing import Any

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.types import TextContent
except ImportError:
    raise

# Logger for MCP call tracing
_mcp_logger = logging.getLogger("telemetry.mcp.calls")


def _csv_row(values: list[str]) -> str:
    """Convert a list of values to a CSV row string."""
    buf = io.StringIO()
    csv.writer(buf).writerow(values)
    return buf.getvalue().strip()


def _approx_tokens(text: str) -> int:
    """Approximate token count using simple word splitting."""
    return len(text.split())


def log_mcp_call(server_name: str, tool_name: str, arguments: dict[str, Any], response: dict[str, Any]):
    """Log an MCP call for tracing."""
    if _mcp_logger.isEnabledFor(logging.INFO):
        resp_text = json.dumps(response, ensure_ascii=False)
        _mcp_logger.info(
            _csv_row(
                [
                    datetime.datetime.now(tz=datetime.timezone.utc).isoformat(timespec="milliseconds") + "Z",
                    server_name,
                    tool_name,
                    json.dumps(arguments, ensure_ascii=False),
                    str(_approx_tokens(resp_text)),
                    str(len(resp_text.encode())),
                ]
            )
        )


@dataclass(slots=True)
class McpServerConfig:
    """Configuration for an MCP server."""

    name: str
    command: str
    args: list[str]
    env: dict[str, str] | None = None
    suppress_output: bool = True  # New field to control output suppression


class McpClient:
    config: McpServerConfig

    def __init__(self, config: McpServerConfig):
        self.server_name = config.name
        self.session: ClientSession | None = None
        self.stdio_context: Any | None = None
        self.session_context: Any | None = None
        self.instructions: str | None = None
        self._connected = False
        self.config = config

    async def connect(self) -> ClientSession:
        """Connect to the configured MCP server."""
        if self._connected:
            if self.session is None:
                raise RuntimeError("Client is marked as connected but session is None")
            return self.session

        # Prepare environment for subprocess
        env = self.config.env.copy() if self.config.env else {}

        # Add environment variable to suppress output if configured
        if self.config.suppress_output:
            env["MCP_SUPPRESS_OUTPUT"] = "1"

        # Create server parameters with modified environment
        server_params = StdioServerParameters(command=self.config.command, args=self.config.args, env=env)

        try:
            self.stdio_context = stdio_client(server_params)
            read, write = await self.stdio_context.__aenter__()

            self.session_context = ClientSession(read, write)
            self.session = await self.session_context.__aenter__()

            # Initialize the session
            if self.session is not None:
                result = await self.session.initialize()
                self.instructions = result.instructions if result.instructions else ""
                self._connected = True
                return self.session
            else:
                raise RuntimeError("Failed to establish MCP session")
        except Exception as e:
            # Clean up any partially created contexts if connection fails
            await self._cleanup_contexts()
            raise RuntimeError(f"Failed to connect to MCP server '{self.config.name}': {e}") from e

    async def disconnect(self):
        """Disconnect from the MCP server."""
        if not self._connected:
            return

        await self._cleanup_contexts()
        self._connected = False

    async def _cleanup_contexts(self):
        """Clean up stdio and session contexts safely."""
        if self.session_context:
            try:
                await self.session_context.__aexit__(None, None, None)
            except Exception:
                pass  # Ignore cleanup errors
            finally:
                self.session_context = None

        if self.stdio_context:
            try:
                await self.stdio_context.__aexit__(None, None, None)
            except Exception:
                pass  # Ignore cleanup errors
            finally:
                self.stdio_context = None

        self.instructions = None
        self.session = None

    @property
    def name(self) -> str:
        """Get the server name this client is bound to."""
        return self.server_name

    async def list_tools(self) -> list[str]:
        """List available tools from the server."""
        if not self._connected or not self.session:
            return []

        tools_result = await self.session.list_tools()
        return [tool.name for tool in tools_result.tools]

    async def get_tool_specs(self) -> list[dict[str, Any]]:
        """Get detailed tool specifications from the server."""
        if not self._connected or not self.session:
            return []

        tools_result = await self.session.list_tools()

        tool_specs = []
        for tool in tools_result.tools:
            # Format in OpenAI tool calling format
            tool_spec = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or f"Tool {tool.name}",
                    "parameters": tool.inputSchema or {"type": "object", "properties": {}, "required": []},
                },
            }
            tool_specs.append(tool_spec)

        return tool_specs

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call a tool on the server."""
        if not self._connected or not self.session:
            raise RuntimeError(f"No active MCP session for server: {self.server_name}")

        result = await self.session.call_tool(tool_name, arguments=arguments)
        final_result = None

        # First, check if there's structured content (preferred)
        if hasattr(result, "structuredContent") and result.structuredContent:
            final_result = {"result": result.structuredContent.get("result", result.structuredContent), "success": True}

        # Fall back to parsing text content
        elif result.content and len(result.content) > 0:
            # If there are multiple content items, try to parse each as JSON and combine
            if len(result.content) > 1:
                parsed_items = []
                for content_item in result.content:
                    if isinstance(content_item, TextContent):
                        try:
                            parsed_item = json.loads(content_item.text)
                            parsed_items.append(parsed_item)
                        except json.JSONDecodeError:
                            # If parsing fails, treat as text
                            parsed_items.append(content_item.text)

                # Return the array of parsed items
                final_result = {"result": parsed_items, "success": True}

            else:
                # Single content item - parse as before
                content_item = result.content[0]
                if isinstance(content_item, TextContent):
                    try:
                        # Try to parse as JSON first
                        parsed_result = json.loads(content_item.text)

                        # If the result is a list, wrap it in a dictionary for consistent handling
                        if isinstance(parsed_result, list):
                            final_result = {"result": parsed_result, "success": True}
                        # If it's already a dict, ensure it has success flag
                        elif isinstance(parsed_result, dict):
                            if "success" not in parsed_result:
                                parsed_result["success"] = True
                            final_result = parsed_result
                        else:
                            # For other types, wrap in a dictionary
                            final_result = {"result": parsed_result, "success": True}

                    except json.JSONDecodeError:
                        # If not JSON, return as plain text result
                        final_result = {"content": content_item.text, "success": True}

        if final_result is None:
            final_result = {"success": False, "error": "No response from server"}

        log_mcp_call(self.server_name, tool_name, arguments, final_result)

        return final_result

    async def is_connected(self) -> bool:
        """Check if the MCP server is connected."""
        return self._connected


class McpClientRegistry:
    """Registry for discovering and managing MCP server configurations."""

    def __init__(self):
        self._servers: dict[str, McpClient] = {}

    def register_server(self, config: McpServerConfig):
        """Register a server configuration."""
        self._servers[config.name] = McpClient(config)

    def get_client(self, name: str) -> McpClient | None:
        """Get a server configuration by name."""
        return self._servers.get(name)

    def list_servers(self) -> list[str]:
        """List all registered server names."""
        return list(self._servers.keys())

    @classmethod
    def from_dict(cls, mcp_servers: dict[str, dict]) -> McpClientRegistry:
        """Create a registry from a dictionary of server configurations."""
        registry = cls()
        for name, config in mcp_servers.items():
            server_config = McpServerConfig(
                name=name,
                command=config.get("command", ""),
                args=config.get("args", []),
                env=config.get("env", {}),
                suppress_output=config.get("suppress_output", True),  # Default to suppressing output
            )
            registry.register_server(server_config)
        return registry

    @classmethod
    def from_config(cls, configs: dict[str, McpServerConfig]) -> McpClientRegistry:
        """Create a registry from a list of server configurations."""
        registry = cls()
        for name, config in configs.items():
            config.name = name
            registry.register_server(config)
        return registry
