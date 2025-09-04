import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from lev.core.mcp import McpClientRegistry, ServerConfig
from lev.core.provider_registry import LlmProviderRegistry
from lev.llm_config_loader import LLMConfigLoader
from lev.llm_providers.provider_factory import create_provider


@dataclass(slots=True)
class ModelConfig:
    provider: str
    model: str
    model_parameters: Dict[str, Any]
    persona: Optional[str] = None  # Persona key or direct system prompt


@dataclass(slots=True)
class SuiteConfig:
    solver: Optional[ModelConfig] = None
    asker: Optional[ModelConfig] = None
    judge: Optional[ModelConfig] = None


@dataclass(slots=True)
class ResolvedEvalSuite:
    """Fully resolved evaluation configuration with providers."""

    name: str
    provider_registry: LlmProviderRegistry
    mcps: Dict[str, ServerConfig]
    mcp_registry: McpClientRegistry
    evals: List[Dict[str, Any]]


class DatasetType(str, Enum):
    """Supported evaluation dataset types."""

    JUDGE_EVAL = "judge_eval"
    SCENARIO_EVAL = "scenario_eval"
    MCP_EVAL = "mcp_eval"


def load_dataset(path: str) -> Tuple[DatasetType, List[Dict[str, Any]], List[str]]:
    """
    Load a typed evaluation dataset from JSON file.

    Expected format:
    {
        "type": "judge_eval" | "scenario_eval" | ...,
        "description": "human-readable summary",
        "eval_method": ["critique"] | ["match"] | ["critique", "match"] (optional),
        "data": [...]  // list of cases or scenarios
    }

    Args:
        path: Path to JSON dataset file

    Returns:
        Tuple of (dataset_type, data_list, eval_method)

    Raises:
        FileNotFoundError: If dataset file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
        KeyError: If required keys are missing
        ValueError: If dataset type is unknown
    """
    try:
        with open(path, "r") as f:
            obj = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file '{path}' not found")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in '{path}': {e}", e.doc, e.pos)

    # Validate required keys
    required_keys = {"type", "data"}
    if not required_keys.issubset(obj.keys()):
        missing = required_keys - obj.keys()
        raise KeyError(f"Dataset missing required keys: {missing}")

    # Validate and convert type
    try:
        dataset_type = DatasetType(obj["type"])
    except ValueError:
        valid_types = [t.value for t in DatasetType]
        raise ValueError(f"Unknown dataset type '{obj['type']}'. Valid types: {valid_types}")

    # Validate data is a list
    data = obj["data"]
    if not isinstance(data, list):
        raise ValueError(f"Dataset 'data' must be a list, got {type(data).__name__}")

    # Extract eval_method with default to ["critique"]
    eval_method = obj.get("eval_method", ["critique"])
    if not isinstance(eval_method, list):
        raise ValueError(f"Dataset 'eval_method' must be a list, got {type(eval_method).__name__}")

    return dataset_type, data, eval_method


def _parse_model_config(config_data: Dict[str, Any], config_name: str) -> ModelConfig:
    """Parse and validate a model configuration from dataset JSON."""
    if not isinstance(config_data, dict):
        raise ValueError(f"Dataset '{config_name}' must be a dict, got {type(config_data).__name__}")

    required_config_fields = {"provider", "model"}
    if not required_config_fields.issubset(config_data.keys()):
        missing = required_config_fields - config_data.keys()
        raise ValueError(f"Dataset {config_name} missing required fields: {missing}")

    return ModelConfig(
        provider=config_data["provider"],
        model=config_data["model"],
        model_parameters=config_data.get("model_parameters", {}),
        persona=config_data.get("persona"),
    )


def load_judge_cases(path: str = "simple_eval_dataset.json") -> List[Dict[str, Any]]:
    """
    Load judge evaluation cases from JSON file.

    Args:
        path: Path to judge evaluation dataset

    Returns:
        List of evaluation cases

    Raises:
        AssertionError: If dataset is not of type JUDGE_EVAL
    """
    dataset_type, data, _ = load_dataset(path)
    if dataset_type != DatasetType.JUDGE_EVAL:
        raise ValueError(f"Expected judge_eval dataset, got {dataset_type}")
    return data


def load_scenarios(path: str = "scenarios.json") -> List[Dict[str, Any]]:
    """
    Load scenario evaluation cases from JSON file.

    Args:
        path: Path to scenario evaluation dataset

    Returns:
        List of scenarios

    Raises:
        AssertionError: If dataset is not of type SCENARIO_EVAL
    """
    dataset_type, data, _ = load_dataset(path)
    if dataset_type != DatasetType.SCENARIO_EVAL:
        raise ValueError(f"Expected scenario_eval dataset, got {dataset_type}")
    return data


def load_scenarios_with_eval_method(path: str = "scenarios.json") -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Load scenario evaluation cases from JSON file with eval_method.

    Args:
        path: Path to scenario evaluation dataset

    Returns:
        Tuple of (scenarios, eval_method)

    Raises:
        AssertionError: If dataset is not of type SCENARIO_EVAL
    """
    dataset_type, data, eval_method = load_dataset(path)
    if dataset_type != DatasetType.SCENARIO_EVAL:
        raise ValueError(f"Expected scenario_eval dataset, got {dataset_type}")
    return data, eval_method


def load_personas(path: str = "personas.json") -> Dict[str, Dict[str, str]]:
    """
    Load persona definitions from JSON file.

    Args:
        path: Path to personas JSON file

    Returns:
        Dictionary mapping persona keys to persona definitions

    Raises:
        FileNotFoundError: If personas file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
    """
    try:
        with open(path, "r") as f:
            personas = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Personas file '{path}' not found")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in '{path}': {e}", e.doc, e.pos)

    if not isinstance(personas, dict):
        raise ValueError(f"Personas file must contain a JSON object, got {type(personas).__name__}")

    return personas


def get_persona_system_prompt(persona_key: str, personas_path: str = "personas.json") -> str:
    """
    Get the system prompt for a specific persona.

    Args:
        persona_key: The key for the persona (e.g., "math_phd", "car_salesman")
        personas_path: Path to personas JSON file

    Returns:
        The system prompt string for that persona

    Raises:
        KeyError: If persona_key is not found
        FileNotFoundError: If personas file doesn't exist
    """
    personas = load_personas(personas_path)

    if persona_key not in personas:
        available = list(personas.keys())
        raise KeyError(f"Unknown persona '{persona_key}'. Available personas: {available}")

    return personas[persona_key]["system_prompt"]


def resolve_persona_system_prompt(persona: Optional[str], personas_path: str = "personas.json") -> Optional[str]:
    """
    Resolve a persona reference to its system prompt.

    Args:
        persona: Persona key (to lookup in personas.json) or direct system prompt string, or None
        personas_path: Path to personas JSON file

    Returns:
        System prompt string if persona is provided, None otherwise

    Raises:
        KeyError: If persona_key is not found in personas file
        FileNotFoundError: If personas file doesn't exist when persona is a key
    """
    if persona is None:
        return None

    # Try to load personas and check if persona is a key
    try:
        personas = load_personas(personas_path)
        if persona in personas:
            return personas[persona]["system_prompt"]
    except (FileNotFoundError, json.JSONDecodeError):
        # If personas file doesn't exist or is invalid, treat persona as direct prompt
        pass

    # If not found in personas or personas file doesn't exist, treat as direct system prompt
    return persona


def load_mcp_dataset(
    path: str = "mcp_eval_dataset.json",
) -> Tuple[List[Dict[str, Any]], Dict[str, ServerConfig], SuiteConfig]:
    """
    Load MCP evaluation dataset from JSON file.

    Args:
        path: Path to MCP evaluation dataset

    Returns:
        Tuple of (scenarios, mcp_servers, scenario_config)

    Raises:
        ValueError: If dataset is not of type MCP_EVAL
    """
    dataset_type, data, _ = load_dataset(path)
    if dataset_type != DatasetType.MCP_EVAL:
        raise ValueError(f"Expected mcp_eval dataset, got {dataset_type}")

    # Load the full dataset to get mcp_servers definition
    try:
        with open(path, "r") as f:
            full_dataset = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise ValueError(f"Failed to load dataset file '{path}': {e}")

    # Extract mcp_servers definition (optional for backward compatibility)
    mcp_servers = full_dataset.get("mcp_servers", {})

    # Extract model configurations for different roles
    scenario_config = SuiteConfig()

    # Check for solver configuration
    if "solver" in full_dataset:
        scenario_config.solver = _parse_model_config(full_dataset["solver"], "solver")

    # Check for asker configuration (optional)
    if "asker" in full_dataset:
        scenario_config.asker = _parse_model_config(full_dataset["asker"], "asker")

    # Check for judge configuration (optional)
    if "judge" in full_dataset:
        scenario_config.judge = _parse_model_config(full_dataset["judge"], "judge")

    # Legacy support: check for old "model_config" field
    if "model_config" in full_dataset and scenario_config.solver is None:
        scenario_config.solver = _parse_model_config(full_dataset["model_config"], "model_config")

    # Validate required fields for each scenario
    required_fields = {"id", "question", "mcps"}
    for i, scenario in enumerate(data):
        if not isinstance(scenario, dict):
            raise ValueError(f"Scenario {i} must be a dict, got {type(scenario).__name__}")

        missing_fields = required_fields - scenario.keys()
        if missing_fields:
            raise ValueError(
                f"Scenario {i} (id: {scenario.get('id', 'unknown')}) missing required fields: {missing_fields}"
            )

        # Validate mcps is a list
        if not isinstance(scenario["mcps"], list):
            raise ValueError(
                f"Scenario {i} 'mcps' must be a list, got {type(scenario['mcps']).__name__}"
            )

        # Validate that all mcps exist in mcp_servers (if mcp_servers defined)
        if mcp_servers:
            mcps = set(scenario["mcps"])
            available_mcps = set(mcp_servers.keys())
            missing_mcps = mcps - available_mcps
            if missing_mcps:
                raise ValueError(
                    f"Scenario {i} (id: {scenario.get('id', 'unknown')}) references undefined MCP servers: {missing_mcps}. "
                    f"Available servers: {list(available_mcps)}"
                )

        # Inject only the allowed mcp_servers configs into this scenario
        scenario_mcp_servers = {}
        for mcp_name in scenario["mcps"]:
            if mcp_name in mcp_servers:
                scenario_mcp_servers[mcp_name] = mcp_servers[mcp_name]

        scenario["mcp_servers"] = scenario_mcp_servers

    return data, mcp_servers, scenario_config


def load_eval_with_mcps(path: str) -> ResolvedEvalSuite:
    """
    Load MCP evaluation dataset with resolved LLM configurations.

    Args:
        path: Path to MCP evaluation dataset

    Returns:
        ResolvedEvalSuite with providers, MCPs, and eval items

    Raises:
        ValueError: If dataset is not of type MCP_EVAL or config is invalid
    """
    # Load the full dataset
    try:
        with open(path, "r") as f:
            full_dataset = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise ValueError(f"Failed to load dataset file '{path}': {e}")

    name = Path(path).stem
    # Validate dataset type
    dataset_type, data, _ = load_dataset(path)
    if dataset_type != DatasetType.MCP_EVAL:
        raise ValueError(f"Expected mcp_eval dataset, got {dataset_type}")

    # Extract mcp_servers definition
    mcp_servers = full_dataset.get("mcp_servers", {})

    # Configure MCP call logging if enabled
    _configure_telemetry_logging(full_dataset, name)

    # Check if we have new llm_config section
    llm_config_data = full_dataset.get("llm_config")
    providers = {}

    if llm_config_data:
        # Use new configuration system
        loader = LLMConfigLoader()

        # Determine which roles we need providers for
        roles_needed = ["solver"]  # Always need solver
        if "overrides" in llm_config_data:
            roles_needed.extend(llm_config_data["overrides"].keys())

        # Create providers for each role
        for role in roles_needed:
            resolved_config = loader.get_llm_config(llm_config_data, role)

            # Convert resolved config to provider kwargs
            provider_kwargs = {}
            if resolved_config.api_key:
                provider_kwargs["api_key"] = resolved_config.api_key
            if resolved_config.endpoint:
                provider_kwargs["endpoint"] = resolved_config.endpoint
            if resolved_config.api_version:
                provider_kwargs["api_version"] = resolved_config.api_version
            if resolved_config.base_url:
                provider_kwargs["base_url"] = resolved_config.base_url
            if resolved_config.region:
                provider_kwargs["region"] = resolved_config.region

            # Add model and parameters
            provider_kwargs["model"] = resolved_config.model
            provider_kwargs.update(resolved_config.model_parameters)

            # Create provider
            providers[role] = create_provider(provider_name=resolved_config.provider, **provider_kwargs)
    else:
        # Fall back to legacy configuration parsing
        scenario_config = SuiteConfig()

        # Check for solver configuration
        if "solver" in full_dataset:
            scenario_config.solver = _parse_model_config(full_dataset["solver"], "solver")

        # Check for asker configuration (optional)
        if "asker" in full_dataset:
            scenario_config.asker = _parse_model_config(full_dataset["asker"], "asker")

        # Check for judge configuration (optional)
        if "judge" in full_dataset:
            scenario_config.judge = _parse_model_config(full_dataset["judge"], "judge")

        # Legacy support: check for old "model_config" field
        if "model_config" in full_dataset and scenario_config.solver is None:
            scenario_config.solver = _parse_model_config(full_dataset["model_config"], "model_config")

        # Create providers from legacy config
        if scenario_config.solver:
            providers["solver"] = create_provider(
                provider_name=scenario_config.solver.provider,
                model=scenario_config.solver.model,
                **scenario_config.solver.model_parameters,
            )

        if scenario_config.asker:
            providers["asker"] = create_provider(
                provider_name=scenario_config.asker.provider,
                model=scenario_config.asker.model,
                **scenario_config.asker.model_parameters,
            )

        if scenario_config.judge:
            providers["judge"] = create_provider(
                provider_name=scenario_config.judge.provider,
                model=scenario_config.judge.model,
                **scenario_config.judge.model_parameters,
            )

    # Validate required fields for each scenario
    required_fields = {"id", "question", "mcps"}
    for i, scenario in enumerate(data):
        if not isinstance(scenario, dict):
            raise ValueError(f"Scenario {i} must be a dict, got {type(scenario).__name__}")

        missing_fields = required_fields - scenario.keys()
        if missing_fields:
            raise ValueError(
                f"Scenario {i} (id: {scenario.get('id', 'unknown')}) missing required fields: {missing_fields}"
            )

        # Validate mcps is a list
        if not isinstance(scenario["mcps"], list):
            raise ValueError(
                f"Scenario {i} 'mcps' must be a list, got {type(scenario['mcps']).__name__}"
            )

        # Validate that all mcps exist in mcp_servers (if mcp_servers defined)
        if mcp_servers:
            mcps = set(scenario["mcps"])
            available_mcps = set(mcp_servers.keys())
            missing_mcps = mcps - available_mcps
            if missing_mcps:
                raise ValueError(
                    f"Scenario {i} (id: {scenario.get('id', 'unknown')}) references undefined MCP servers: {missing_mcps}. "
                    f"Available servers: {list(available_mcps)}"
                )

        # Inject only the allowed mcp_servers configs into this scenario
        scenario_mcp_servers = {}
        for mcp_name in scenario["mcps"]:
            if mcp_name in mcp_servers:
                scenario_mcp_servers[mcp_name] = mcp_servers[mcp_name]

        scenario["mcp_servers"] = scenario_mcp_servers

    # Create LLM provider registry
    provider_registry = LlmProviderRegistry(_providers=providers)

    # Create MCP registry from server configurations
    mcp_registry = McpClientRegistry.from_dict(mcp_servers)

    return ResolvedEvalSuite(
        name=name, provider_registry=provider_registry, mcps=mcp_servers, mcp_registry=mcp_registry, evals=data
    )


def _configure_telemetry_logging(dataset: Dict[str, Any], suite_name: str) -> None:
    """
    Configure telemetry logging based on dataset configuration.

    Args:
        dataset: The full dataset configuration
        suite_name: Base name for the suite (used for log file naming)
    """
    # Check for logging configuration
    logging_config = dataset.get("logging", {})
    mcp_calls_enabled = logging_config.get("mcp_calls", False)

    # Get the MCP logger
    mcp_logger = logging.getLogger("telemetry.mcp.calls")

    if mcp_calls_enabled:
        # Set up file logging for MCP calls
        log_path = f"{suite_name}_mcp_log.csv"

        # Remove any existing handlers to avoid duplicates
        for handler in mcp_logger.handlers[:]:
            mcp_logger.removeHandler(handler)

        # Create file handler
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(message)s"))

        # Configure logger
        mcp_logger.setLevel(logging.INFO)
        mcp_logger.addHandler(file_handler)
        mcp_logger.propagate = False  # Prevent propagation to root logger

        # Write CSV header if file is new/empty
        try:
            if Path(log_path).stat().st_size == 0:
                mcp_logger.info("timestamp,server_name,tool_name,arguments,response_size_tokens,response_size_bytes")
        except (FileNotFoundError, OSError):
            # File doesn't exist yet, header will be written on first log
            mcp_logger.info("timestamp,server_name,tool_name,arguments,response_size_tokens,response_size_bytes")
    else:
        # Disable MCP call logging
        mcp_logger.setLevel(logging.CRITICAL)
        # Add null handler to avoid "No handler" warnings
        if not mcp_logger.handlers:
            mcp_logger.addHandler(logging.NullHandler())
        mcp_logger.propagate = False


def validate_mcp_usage(scenario: Dict[str, Any], mcps: List[str]) -> bool:
    """
    Validate that only allowed MCPs were used in a scenario.

    Args:
        scenario: The scenario definition with 'mcps' field
        mcps: List of MCP server names that were actually used

    Returns:
        True if all used MCPs are allowed, False otherwise
    """
    allowed = set(scenario.get("mcps", []))
    used = set(mcps)

    # Check if any disallowed MCPs were used
    disallowed_usage = used - allowed
    return len(disallowed_usage) == 0
