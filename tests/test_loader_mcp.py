"""
Tests for load_manifest functionality with black box testing approach.
"""

import json
import os
from unittest.mock import mock_open, patch

import pytest

from lev.llm_config_loader import LLMConfigLoader, ModelMapping, ProviderProfile, ProviderType
from lev.loader import load_manifest
from lev.manifest import DatasetType, EvalManifest, ResolvedEvalManifest


class TestLoadManifest:
    """Test suite for load_manifest using black box testing with mocked file access."""

    @pytest.fixture
    def mock_profiles(self):
        """Create mock profiles for testing."""
        return {
            "azure_openai": ProviderProfile(
                provider=ProviderType.AZURE_OPENAI,
                models=ModelMapping(default="gpt-4", reasoning="gpt-4o", fast="gpt-3.5-turbo"),
                api_key_env="AZURE_OPENAI_API_KEY",
                endpoint_env="AZURE_OPENAI_ENDPOINT",
                api_version="2024-02-15-preview",
            ),
            "test_profile": ProviderProfile(
                provider=ProviderType.OPENAI,
                models=ModelMapping(default="gpt-4", reasoning="gpt-4o", fast="gpt-3.5-turbo"),
                api_key_env="OPENAI_API_KEY",
            ),
        }

    @pytest.fixture
    def mock_llm_config_loader(self, mock_profiles):
        """Create a mock LLMConfigLoader with test profiles."""

        def mock_init(self, profiles_path=None):
            self.profiles = mock_profiles

        # Patch the __init__ method to use our mock profiles
        with patch.object(LLMConfigLoader, "__init__", mock_init):
            yield LLMConfigLoader

    @pytest.fixture
    def mock_env_vars(self):
        """Mock environment variables for API keys."""
        env_vars = {
            "AZURE_OPENAI_API_KEY": "test-azure-key",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
            "OPENAI_API_KEY": "test-openai-key",
        }
        with patch.dict(os.environ, env_vars):
            yield

    @pytest.fixture
    def valid_manifest_json(self):
        """Valid manifest data with mixed eval formats as JSON string."""
        data = {
            "schema_version": 1,
            "type": "mcp_eval",
            "description": "Test MCP evaluation",
            "logging": {"mcp_calls": True, "results": True},
            "llm_config": {
                "active_profile": "azure_openai",
                "overrides": {
                    "solver": {"persona": "concise_solver"},
                    "solver.reasoning": {"model_variant": "reasoning"},
                    "asker": {"model_variant": "fast", "persona": "diligent_asker"},
                },
            },
            "mcp_servers": {
                "test-mcp": {"command": "python", "args": ["test_mcp.py"], "env": {"TEST_VAR": "test_value"}}
            },
            "evals": [
                {
                    "id": "new_format_test",
                    "question": "Test question with new format",
                    "execution": {
                        "mcps": ["test-mcp"],
                        "solver": {"max_reasoning_steps": 4, "max_retrospective_turns": 1},
                        "asker": {"max_turns": 1},
                    },
                    "expectations": {
                        "tool_calls": ["test_tool"],
                        "tool_inputs": {"test_tool": {"acceptable_queries": []}},
                        "results": {
                            "test_tool": {
                                "test_record": {"schema": [{"id": "int", "value": "string"}], "values": [[1, "test"]]}
                            }
                        },
                    },
                    "scoring": [{"type": "llm_judge", "mode": "critique"}],
                },
                {
                    "id": "legacy_format_test",
                    "question": "Test question with legacy format",
                    "asker_turns": 1,
                    "mcps": ["test-mcp"],
                    "expectations": {
                        "tool_calls": ["test_tool"],
                        "tool_inputs": {"test_tool": {"acceptable_queries": []}},
                        "results": {
                            "test_tool": {
                                "test_record": {"schema": [{"id": "int", "value": "string"}], "values": [[2, "legacy"]]}
                            }
                        },
                    },
                    "scoring": ["critique"],
                },
            ],
        }
        return json.dumps(data)

    @pytest.fixture
    def minimal_manifest_json(self):
        """Minimal valid manifest data as JSON string."""
        data = {
            "schema_version": 1,
            "type": "mcp_eval",
            "description": "Minimal test",
            "llm_config": {"active_profile": "test_profile"},
            "mcp_servers": {},
            "evals": [],
        }
        return json.dumps(data)

    @pytest.fixture
    def invalid_type_manifest_json(self):
        """Manifest with invalid type as JSON string."""
        data = {
            "schema_version": 1,
            "type": "regular_eval",  # Wrong type
            "description": "Invalid type test",
            "llm_config": {"active_profile": "test_profile"},
            "mcp_servers": {},
            "evals": [],
        }
        return json.dumps(data)

    @patch("builtins.open", new_callable=mock_open)
    def test_load_manifest_success(self, mock_file, valid_manifest_json, mock_llm_config_loader, mock_env_vars):
        """Test successfully loading a valid manifest file."""
        mock_file.return_value.read.return_value = valid_manifest_json

        result = load_manifest("test_manifest.json")

        # Verify the manifest file was opened (allow for additional file operations like CSV logging)
        manifest_calls = [call for call in mock_file.call_args_list if call[0][0] == "test_manifest.json"]
        assert len(manifest_calls) == 1
        assert manifest_calls[0] == (("test_manifest.json", "r"),)

        # Verify result structure (ResolvedEvalManifest)
        assert isinstance(result, ResolvedEvalManifest)
        assert result.name == "test_manifest"
        assert result.provider_registry is not None
        assert result.mcp_registry is not None
        assert len(result.evals) == 2
        assert result.evals[0].id == "new_format_test"
        assert result.evals[1].id == "legacy_format_test"
        assert result.result_sink is not None  # Should have result sink due to logging config

    @patch("builtins.open", side_effect=FileNotFoundError("File not found"))
    def test_load_manifest_file_not_found(self, mock_file):
        """Test handling of missing manifest file."""
        with pytest.raises(FileNotFoundError, match="Manifest file 'missing.json' not found"):
            load_manifest("missing.json")

    @patch("builtins.open", new_callable=mock_open)
    def test_load_manifest_invalid_json(self, mock_file):
        """Test handling of invalid JSON in manifest file."""
        mock_file.return_value.read.return_value = "{ invalid json }"

        with pytest.raises(json.JSONDecodeError, match="Invalid JSON in 'invalid.json'"):
            load_manifest("invalid.json")

    @patch("builtins.open", new_callable=mock_open)
    def test_load_manifest_minimal_config(
        self, mock_file, minimal_manifest_json, mock_llm_config_loader, mock_env_vars
    ):
        """Test loading manifest with minimal configuration."""
        mock_file.return_value.read.return_value = minimal_manifest_json

        result = load_manifest("minimal_manifest.json")

        assert isinstance(result, ResolvedEvalManifest)
        assert result.name == "minimal_manifest"
        assert result.provider_registry is not None
        assert result.mcp_registry is not None
        assert len(result.evals) == 0
        assert result.result_sink is None  # No logging config

    @patch("builtins.open", new_callable=mock_open)
    def test_load_manifest_wrong_dataset_type(
        self, mock_file, invalid_type_manifest_json, mock_llm_config_loader, mock_env_vars
    ):
        """Test handling of wrong dataset type."""
        mock_file.return_value.read.return_value = invalid_type_manifest_json

        with pytest.raises(ValueError, match="Expected mcp_eval dataset, got regular_eval"):
            load_manifest("invalid_type_manifest.json")

    @pytest.fixture
    def multi_role_manifest_json(self):
        """Manifest with multiple roles including dotted notation."""
        data = {
            "schema_version": 1,
            "type": "mcp_eval",
            "description": "Multi-role test",
            "llm_config": {
                "active_profile": "test_profile",
                "overrides": {
                    "solver": {"persona": "base_solver"},
                    "solver.reasoning": {"model_variant": "reasoning", "persona": "reasoning_solver"},
                    "asker": {"model_variant": "fast", "persona": "fast_asker"},
                    "judge": {"model_variant": "default", "persona": "strict_judge"},
                },
            },
            "mcp_servers": {},
            "evals": [],
        }
        return json.dumps(data)

    @patch("builtins.open", new_callable=mock_open)
    def test_load_manifest_multi_role_providers(
        self, mock_file, multi_role_manifest_json, mock_llm_config_loader, mock_env_vars
    ):
        """Test that providers are created for all roles mentioned in overrides."""
        mock_file.return_value.read.return_value = multi_role_manifest_json

        result = load_manifest("multi_role_manifest.json")

        # Should have providers for solver, asker, and judge (base roles from overrides)
        assert result.provider_registry is not None
        assert result.provider_registry.has_role("solver")
        assert result.provider_registry.has_role("asker")
        assert result.provider_registry.has_role("judge")

    @pytest.fixture
    def no_solver_manifest_json(self):
        """Manifest without solver role."""
        data = {
            "schema_version": 1,
            "type": "mcp_eval",
            "description": "No solver test",
            "llm_config": {
                "active_profile": "test_profile",
                "overrides": {
                    "asker": {"model_variant": "fast"},
                },
            },
            "mcp_servers": {},
            "evals": [],
        }
        return json.dumps(data)

    @patch("builtins.open", new_callable=mock_open)
    def test_load_manifest_missing_solver_fails(
        self, mock_file, no_solver_manifest_json, mock_llm_config_loader, mock_env_vars
    ):
        """Test that manifest without solver overrides still creates solver provider."""
        mock_file.return_value.read.return_value = no_solver_manifest_json

        # This should succeed because solver provider is always created with defaults
        result = load_manifest("no_solver_manifest.json")
        assert result.provider_registry.has_role("solver")
        assert result.provider_registry.has_role("asker")

    @pytest.fixture
    def empty_llm_config_manifest_json(self):
        """Manifest with no llm_config section."""
        data = {
            "schema_version": 1,
            "type": "mcp_eval",
            "description": "Empty LLM config test",
            "mcp_servers": {},
            "evals": [],
        }
        return json.dumps(data)

    @patch("builtins.open", new_callable=mock_open)
    def test_load_manifest_empty_llm_config_fails(
        self, mock_file, empty_llm_config_manifest_json, mock_llm_config_loader, mock_env_vars
    ):
        """Test that missing llm_config causes validation error."""
        mock_file.return_value.read.return_value = empty_llm_config_manifest_json

        with pytest.raises(ValueError, match="No solver provider configured. The 'solver' role is required."):
            load_manifest("empty_llm_config_manifest.json")


if __name__ == "__main__":
    pytest.main([__file__])
