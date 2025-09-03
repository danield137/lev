import json
import os
import tempfile

import pytest

from lev.core.mcp import McpClientRegistry, ServerConfig
from lev.dataset_loader import load_mcp_dataset, validate_mcp_usage


class TestMcpDatasetLoader:
    """Test MCP dataset loading functionality."""

    def test_load_mcp_dataset_success(self):
        """Test successful loading of MCP dataset."""
        # Create temporary dataset file with envelope format
        test_data = {
            "type": "mcp_eval",
            "description": "Test dataset",
            "mcp_servers": {"filesystem-mcp": {"command": "python", "args": ["lev/samples/fs_mcp.py"]}},
            "data": [
                {
                    "id": "test_scenario",
                    "question": "Test question?",
                    "asker": "default",
                    "solver": "default",
                    "allowed_mcps": ["filesystem-mcp"],
                    "expected_tool_calls": ["read_file"],
                }
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name

        try:
            scenarios, mcp_servers, _ = load_mcp_dataset(temp_path)
            assert len(scenarios) == 1
            assert scenarios[0]["id"] == "test_scenario"
            assert scenarios[0]["allowed_mcps"] == ["filesystem-mcp"]
            # Check that mcp_servers config was injected
            assert "mcp_servers" in scenarios[0]
            assert "filesystem-mcp" in scenarios[0]["mcp_servers"]
            assert scenarios[0]["mcp_servers"]["filesystem-mcp"]["command"] == "python"
            # Check mcp_servers return value
            assert "filesystem-mcp" in mcp_servers
            assert mcp_servers["filesystem-mcp"]["command"] == "python"
        finally:
            os.unlink(temp_path)

    def test_load_mcp_dataset_missing_file(self):
        """Test loading non-existent dataset file."""
        with pytest.raises(FileNotFoundError):
            load_mcp_dataset("nonexistent.json")

    def test_load_mcp_dataset_invalid_json(self):
        """Test loading invalid JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json content")
            temp_path = f.name

        try:
            with pytest.raises(json.JSONDecodeError):
                load_mcp_dataset(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_mcp_dataset_missing_required_fields(self):
        """Test loading dataset with missing required fields."""
        test_data = {
            "type": "mcp_eval",
            "description": "Test dataset",
            "data": [
                {
                    "id": "test_scenario",
                    "question": "Test question?",
                    # Missing allowed_mcps field
                }
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="missing required fields"):
                load_mcp_dataset(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_mcp_dataset_invalid_allowed_mcps_type(self):
        """Test loading dataset with invalid allowed_mcps type."""
        test_data = {
            "type": "mcp_eval",
            "description": "Test dataset",
            "data": [
                {"id": "test_scenario", "question": "Test question?", "allowed_mcps": "not-a-list"}  # Should be a list
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="must be a list"):
                load_mcp_dataset(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_mcp_dataset_undefined_mcp_server(self):
        """Test loading dataset where scenario references undefined MCP server."""
        test_data = {
            "type": "mcp_eval",
            "description": "Test dataset",
            "mcp_servers": {"filesystem-mcp": {"command": "python", "args": ["lev/samples/fs_mcp.py"]}},
            "data": [
                {
                    "id": "test_scenario",
                    "question": "Test question?",
                    "allowed_mcps": ["filesystem-mcp", "undefined-mcp"],  # undefined-mcp not in mcp_servers
                }
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="references undefined MCP servers"):
                load_mcp_dataset(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_mcp_dataset_backward_compatibility(self):
        """Test loading dataset without mcp_servers for backward compatibility."""
        test_data = {
            "type": "mcp_eval",
            "description": "Test dataset",
            # No mcp_servers section for backward compatibility
            "data": [
                {
                    "id": "test_scenario",
                    "question": "Test question?",
                    "allowed_mcps": ["filesystem-mcp"],
                }
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name

        try:
            scenarios, mcp_servers, _ = load_mcp_dataset(temp_path)
            assert len(scenarios) == 1
            assert scenarios[0]["id"] == "test_scenario"
            assert scenarios[0]["allowed_mcps"] == ["filesystem-mcp"]
            # Should have empty mcp_servers config injected
            assert "mcp_servers" in scenarios[0]
            assert scenarios[0]["mcp_servers"] == {}
            assert mcp_servers == {}
        finally:
            os.unlink(temp_path)


class TestMcpValidation:
    """Test MCP usage validation functionality."""

    def test_validate_mcp_usage_valid(self):
        """Test valid MCP usage."""
        scenario = {"allowed_mcps": ["filesystem-mcp", "weather-mcp"]}
        used_mcps = ["filesystem-mcp"]

        assert validate_mcp_usage(scenario, used_mcps) is True

    def test_validate_mcp_usage_invalid(self):
        """Test invalid MCP usage."""
        scenario = {"allowed_mcps": ["filesystem-mcp"]}
        used_mcps = ["filesystem-mcp", "unauthorized-mcp"]

        assert validate_mcp_usage(scenario, used_mcps) is False

    def test_validate_mcp_usage_empty_allowed(self):
        """Test validation with empty allowed MCPs."""
        scenario = {"allowed_mcps": []}
        used_mcps = ["any-mcp"]

        assert validate_mcp_usage(scenario, used_mcps) is False

    def test_validate_mcp_usage_no_usage(self):
        """Test validation with no MCPs used."""
        scenario = {"allowed_mcps": ["filesystem-mcp"]}
        used_mcps = []

        assert validate_mcp_usage(scenario, used_mcps) is True

    def test_validate_mcp_usage_missing_allowed_mcps(self):
        """Test validation with missing allowed_mcps field."""
        scenario = {}  # No allowed_mcps field
        used_mcps = ["some-mcp"]

        assert validate_mcp_usage(scenario, used_mcps) is False


class TestMcpServerConfig:
    """Test MCP server configuration."""

    def test_server_config_creation(self):
        """Test creating server configuration."""
        config = ServerConfig(name="test-server", command="python", args=["test_server.py"], env={"TEST_VAR": "value"})

        assert config.name == "test-server"
        assert config.command == "python"
        assert config.args == ["test_server.py"]
        assert config.env == {"TEST_VAR": "value"}

    def test_registry_registration(self):
        """Test server registry functionality."""
        registry = McpClientRegistry()

        config = ServerConfig(name="filesystem-mcp", command="python", args=["lev/samples/fs_mcp.py"])

        registry.register_server(config)

        retrieved_client = registry.get_client("filesystem-mcp")
        assert retrieved_client is not None
        assert retrieved_client.name == "filesystem-mcp"
        assert retrieved_client.config.command == "python"

    def test_registry_list_servers(self):
        """Test listing registered servers."""
        registry = McpClientRegistry()

        # Should start empty
        servers = registry.list_servers()
        assert len(servers) == 0

        # Add our server
        config = ServerConfig(name="test-server", command="python", args=["test.py"])
        registry.register_server(config)

        servers = registry.list_servers()
        assert "test-server" in servers
        assert len(servers) == 1

    def test_registry_nonexistent_server(self):
        """Test retrieving non-existent server."""
        registry = McpClientRegistry()

        client = registry.get_client("nonexistent-server")
        assert client is None


@pytest.mark.asyncio
class TestMcpClientIntegration:
    """Test MCP client integration (mocked)."""

    async def test_mcp_client_creation(self):
        """Test MCP client creation with registry."""
        registry = McpClientRegistry()
        server_config = ServerConfig(name="test-mcp", command="python", args=["test.py"])
        registry.register_server(server_config)

        client = registry.get_client("test-mcp")

        assert client.server_name == "test-mcp"
        assert client.name == "test-mcp"
        assert not await client.is_connected()

    async def test_mcp_client_server_binding(self):
        """Test that client is bound to specific server."""
        registry = McpClientRegistry()

        config = ServerConfig(name="test-mcp", command="python", args=["test.py"])
        registry.register_server(config)

        client = registry.get_client("test-mcp")

        assert client.name == "test-mcp"
        assert client.server_name == "test-mcp"

    async def test_mcp_client_not_connected_initially(self):
        """Test that client is not connected initially."""
        registry = McpClientRegistry()
        server_config = ServerConfig(name="test-mcp", command="python", args=["test.py"])
        registry.register_server(server_config)

        client = registry.get_client("test-mcp")

        is_connected = await client.is_connected()
        assert is_connected is False

    async def test_mcp_client_invalid_server(self):
        """Test creating client with non-existent server."""
        registry = McpClientRegistry()

        # This should not raise an error during construction
        client = registry.get_client("test-mcp")
        assert client is None, "Client should be None for non-registered server"


def test_dataset_schema_compliance():
    """Test that the actual MCP dataset file complies with expected schema."""
    try:
        scenarios, mcps, _ = load_mcp_dataset("mcp_eval_dataset.json")

        # Verify we have scenarios
        assert len(scenarios) > 0

        # Check each scenario has required fields
        required_fields = {"id", "question", "allowed_mcps"}
        for scenario in scenarios:
            assert all(field in scenario for field in required_fields)
            assert isinstance(scenario["allowed_mcps"], list)
            assert len(scenario["allowed_mcps"]) > 0

        # Check specific scenarios exist
        scenario_ids = [s["id"] for s in scenarios]
        assert "count_letter_a" in scenario_ids
        assert "summarize_python_files" in scenario_ids

    except FileNotFoundError:
        pytest.skip("mcp_eval_dataset.json not found - test requires dataset file")
