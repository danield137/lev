from __future__ import annotations
from lev.host.mcp_client import McpClient, McpServerConfig


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

    def get_all_clients(self) -> list[McpClient]:
        """Get all registered MCP clients."""
        return list(self._servers.values())

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
