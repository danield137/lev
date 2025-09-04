"""
Tests for LLM configuration loader functionality.
"""

import json
import tempfile
from pathlib import Path

import pytest

from lev.dataset_loader import load_eval_with_mcps
from lev.llm_config_loader import (
    LLMConfig,
    LLMConfigLoader,
    ModelMapping,
    ModelParameters,
    ProviderProfile,
    ProviderType,
    RoleConfig,
)


def test_model_mapping_get_model():
    """Test ModelMapping.get_model with different variants."""
    mapping = ModelMapping(default="gpt-4o", reasoning="o1-preview", fast="gpt-4o-mini")

    assert mapping.get_model("default") == "gpt-4o"
    assert mapping.get_model("reasoning") == "o1-preview"
    assert mapping.get_model("fast") == "gpt-4o-mini"
    assert mapping.get_model("unknown") == "gpt-4o"  # fallback to default


def test_model_mapping_from_dict():
    """Test ModelMapping.from_dict creation."""
    data = {"default": "gpt-4o", "reasoning": "o1-preview", "fast": "gpt-4o-mini"}
    mapping = ModelMapping.from_dict(data)

    assert mapping.default == "gpt-4o"
    assert mapping.reasoning == "o1-preview"
    assert mapping.fast == "gpt-4o-mini"


def test_provider_profile_from_dict():
    """Test ProviderProfile.from_dict creation."""
    data = {
        "provider": "openai",
        "api_key_env": "OPENAI_API_KEY",
        "models": {"default": "gpt-4o", "reasoning": "o1-preview", "fast": "gpt-4o-mini"},
    }
    profile = ProviderProfile.from_dict(data)

    assert profile.provider == ProviderType.OPENAI
    assert profile.api_key_env == "OPENAI_API_KEY"
    assert isinstance(profile.models, ModelMapping)
    assert profile.models.default == "gpt-4o"


def test_llm_config_from_dict():
    """Test LLMConfig.from_dict creation."""
    data = {
        "active_profile": "openai",
        "defaults": {"model_parameters": {"temperature": 0.1}},
        "overrides": {
            "solver": {"persona": "concise_solver"},
            "solver.reasoning": {"model_variant": "reasoning", "model_parameters": {"temperature": 0.0}},
        },
    }

    llm_config = LLMConfig.from_dict(data)

    assert llm_config.active_profile == "openai"
    assert isinstance(llm_config.defaults, RoleConfig)
    assert llm_config.defaults.model_parameters.temperature == 0.1
    assert "solver" in llm_config.overrides
    assert llm_config.overrides["solver"].persona == "concise_solver"


def test_llm_config_loader_with_test_profiles():
    """Test LLMConfigLoader with test provider profiles."""
    # Create temporary profiles file
    profiles_data = {
        "profiles": {
            "test_openai": {
                "provider": "openai",
                "api_key_env": "TEST_OPENAI_API_KEY",
                "models": {"default": "gpt-4o", "reasoning": "o1-preview", "fast": "gpt-4o-mini"},
            },
            "test_lmstudio": {
                "provider": "lmstudio",
                "base_url": "http://localhost:1234",
                "models": {"default": "test-model", "reasoning": "test-model", "fast": "test-model"},
            },
        }
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(profiles_data, f)
        profiles_path = f.name

    try:
        loader = LLMConfigLoader(profiles_path)

        # Test basic loading
        assert "test_openai" in loader.profiles
        assert "test_lmstudio" in loader.profiles

        # Test OpenAI profile
        openai_profile = loader.profiles["test_openai"]
        assert openai_profile.provider == ProviderType.OPENAI
        assert openai_profile.api_key_env == "TEST_OPENAI_API_KEY"

        # Test LMStudio profile
        lmstudio_profile = loader.profiles["test_lmstudio"]
        assert lmstudio_profile.provider == ProviderType.LMSTUDIO
        assert lmstudio_profile.base_url == "http://localhost:1234"

    finally:
        Path(profiles_path).unlink()


def test_llm_config_loader_get_llm_config():
    """Test LLMConfigLoader.get_llm_config resolution."""
    # Create temporary profiles file
    profiles_data = {
        "profiles": {
            "test_lmstudio": {
                "provider": "lmstudio",
                "base_url": "http://localhost:1234",
                "models": {"default": "test-model", "reasoning": "test-reasoning-model", "fast": "test-fast-model"},
            }
        }
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(profiles_data, f)
        profiles_path = f.name

    try:
        loader = LLMConfigLoader(profiles_path)

        llm_config_data = {
            "active_profile": "test_lmstudio",
            "defaults": {"model_parameters": {"temperature": 0.7}},
            "overrides": {
                "solver": {"persona": "concise_solver"},
                "solver.reasoning": {"model_variant": "reasoning", "model_parameters": {"temperature": 0.0}},
            },
        }

        # Test default solver config
        solver_config = loader.get_llm_config(llm_config_data, "solver")
        assert solver_config.provider == "lmstudio"
        assert solver_config.model == "test-model"
        assert solver_config.persona == "concise_solver"
        assert solver_config.model_parameters["temperature"] == 0.7
        assert solver_config.base_url == "http://localhost:1234"

        # Test reasoning solver config
        reasoning_config = loader.get_llm_config(llm_config_data, "solver.reasoning")
        assert reasoning_config.provider == "lmstudio"
        assert reasoning_config.model == "test-reasoning-model"
        assert reasoning_config.model_parameters["temperature"] == 0.0
        assert reasoning_config.base_url == "http://localhost:1234"

    finally:
        Path(profiles_path).unlink()


def test_load_eval_with_mcps_new_config():
    """Test load_eval_with_mcps with new llm_config format."""
    # Create temporary dataset file with new llm_config
    dataset_data = {
        "type": "mcp_eval",
        "description": "Test MCP evaluation",
        "llm_config": {
            "active_profile": "test_lmstudio",
            "defaults": {"model_parameters": {"temperature": 0.1}},
            "overrides": {"solver": {"persona": "concise_solver"}},
        },
        "mcp_servers": {"test-mcp": {"command": "python", "args": ["test_mcp.py"]}},
        "data": [{"id": "test_case", "question": "Test question", "mcps": ["test-mcp"]}],
    }

    # Create temporary profiles file
    profiles_data = {
        "profiles": {
            "test_lmstudio": {
                "provider": "lmstudio",
                "base_url": "http://localhost:1234",
                "models": {"default": "test-model"},
            }
        }
    }

    with (
        tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as dataset_f,
        tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as profiles_f,
    ):

        json.dump(dataset_data, dataset_f)
        json.dump(profiles_data, profiles_f)
        dataset_path = dataset_f.name
        profiles_path = profiles_f.name

    try:
        # Temporarily set environment variable for profiles path
        import os

        old_env = os.environ.get("EVAL_PROFILES_PATH")
        os.environ["EVAL_PROFILES_PATH"] = profiles_path

        try:
            resolved_eval = load_eval_with_mcps(dataset_path)

            # Verify structure
            assert resolved_eval.provider_registry.has_role("solver")
            assert "test-mcp" in resolved_eval.mcps
            assert len(resolved_eval.evals) == 1
            assert resolved_eval.evals[0]["id"] == "test_case"

            # Verify provider was created correctly
            solver_provider = resolved_eval.provider_registry.get_solver()
            assert solver_provider is not None

        finally:
            if old_env is not None:
                os.environ["EVAL_PROFILES_PATH"] = old_env
            else:
                os.environ.pop("EVAL_PROFILES_PATH", None)

    finally:
        Path(dataset_path).unlink()
        Path(profiles_path).unlink()


def test_load_eval_with_mcps_legacy_config():
    """Test load_eval_with_mcps with legacy solver/asker/judge config."""
    # Create temporary dataset file with legacy config
    dataset_data = {
        "type": "mcp_eval",
        "description": "Test MCP evaluation",
        "solver": {"provider": "lmstudio", "model": "test-model", "model_parameters": {"temperature": 0.1}},
        "mcp_servers": {"test-mcp": {"command": "python", "args": ["test_mcp.py"]}},
        "data": [{"id": "test_case", "question": "Test question", "mcps": ["test-mcp"]}],
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(dataset_data, f)
        dataset_path = f.name

    try:
        resolved_eval = load_eval_with_mcps(dataset_path)

        # Verify structure
        assert resolved_eval.provider_registry.has_role("solver")
        assert "test-mcp" in resolved_eval.mcps
        assert len(resolved_eval.evals) == 1
        assert resolved_eval.evals[0]["id"] == "test_case"

        # Verify provider was created correctly
        solver_provider = resolved_eval.provider_registry.get_solver()
        assert solver_provider is not None

    finally:
        Path(dataset_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__])
