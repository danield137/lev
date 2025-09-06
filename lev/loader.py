import json
from pathlib import Path

from lev.common.extensions import from_dict
from lev.host.mcp import McpClientRegistry
from lev.core.provider_registry import LlmProviderRegistry
from lev.results import ResultSink
from lev.llm_config_loader import LLMConfigLoader
from lev.llm_providers.provider_factory import create_provider
from lev.logging import configure_telemetry_logging
from lev.manifest import DatasetType, EvalManifest, ResolvedEvalManifest
from lev.output import create_tsv_result_sink


def load_personas(path: str = "personas.json") -> dict[str, dict[str, str]]:
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
    personas = load_personas(personas_path)

    if persona_key not in personas:
        available = list(personas.keys())
        raise KeyError(f"Unknown persona '{persona_key}'. Available personas: {available}")

    return personas[persona_key]["system_prompt"]


def create_provider_registry(manifest: EvalManifest) -> LlmProviderRegistry:
    llm_config_data = manifest.llm_config
    providers = {}

    if llm_config_data:
        # Use new configuration system
        loader = LLMConfigLoader()

        # Determine which roles we need providers for
        # Start with solver (required) and add all roles mentioned in overrides
        roles_needed = {"solver"}
        if llm_config_data.overrides:
            # Extract base role from dotted notation (e.g., "solver.reasoning" -> "solver")
            for override_role in llm_config_data.overrides.keys():
                base_role = override_role.split(".")[0]
                roles_needed.add(base_role)

        # Create providers for each role
        for role in roles_needed:
            try:
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

                # Add model (but not model_parameters directly to avoid provider factory issues)
                provider_kwargs["model"] = resolved_config.model

                # Only add supported model parameters to avoid factory rejections
                supported_params = ["temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty"]
                for param, value in resolved_config.model_parameters.items():
                    if param in supported_params and value is not None:
                        provider_kwargs[param] = value

                # Create provider
                providers[role] = create_provider(provider_name=resolved_config.provider, **provider_kwargs)
            except Exception as e:
                raise ValueError(f"Failed to create provider for role '{role}': {e}") from e

    # Validate that we have at least a solver provider
    if "solver" not in providers:
        raise ValueError("No solver provider configured. The 'solver' role is required.")

    # Create LLM provider registry
    provider_registry = LlmProviderRegistry(_providers=providers)
    return provider_registry


def load_manifest(path: str) -> ResolvedEvalManifest:
    name = Path(path).stem

    manifest: EvalManifest
    # open file with json loader
    try:
        with open(path, "r") as f:
            manifest_data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Manifest file '{path}' not found")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in '{path}': {e}", e.doc, e.pos)

    manifest = from_dict(EvalManifest, manifest_data)

    if manifest.type != DatasetType.MCP_EVAL:
        raise ValueError(f"Expected mcp_eval dataset, got {manifest.type}")

    # Configure MCP call logging if enabled
    configure_telemetry_logging(manifest, name)

    # Configure results sink if enabled
    result_sink = _configure_results_sink(manifest, name)

    # Check if we have new llm_config section
    provider_registry = create_provider_registry(manifest)

    # Create MCP registry from server configurations
    mcp_registry = McpClientRegistry.from_config(manifest.mcp_servers)

    # Validate evals, make sure each eval's MCPs are in the manifest
    for eval in manifest.evals:
        if eval.execution and eval.execution.mcps:
            for mcp in eval.execution.mcps:
                if mcp not in manifest.mcp_servers:
                    raise ValueError(f"Eval '{eval}' references unknown MCP '{mcp}' not in manifest")

    return ResolvedEvalManifest(
        name=name,
        provider_registry=provider_registry,
        mcps=manifest.mcp_servers,
        mcp_registry=mcp_registry,
        evals=manifest.evals,
        result_sink=result_sink,
    )


def _configure_results_sink(manifest: EvalManifest, manifest_name: str) -> ResultSink | None:
    """
    Configure results sink based on manifest configuration.

    Args:
        manifest: The full manifest configuration
        manifest_name: Base name for the suite (used for file naming)

    Returns:
        ResultSink instance if results logging is enabled, None otherwise
    """
    # Check for logging configuration
    logging_config = manifest.logging or {}
    results_enabled = logging_config.get("results", False)

    if not results_enabled:
        return None

    # Get sink type (default to CSV)
    sink_type = logging_config.get("results_sink", "csv")

    if sink_type == "csv":
        return create_tsv_result_sink(manifest_name)
    else:
        # For future extensibility
        raise ValueError(f"Unknown results sink type: {sink_type}. Supported types: csv")
