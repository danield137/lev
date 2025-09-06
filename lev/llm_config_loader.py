import json
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

# Load environment variables from dotenv file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # dotenv not installed, rely on system environment variables
    pass


class ProviderType(str, Enum):
    """Supported LLM provider types."""

    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    BEDROCK = "bedrock"
    LMSTUDIO = "lmstudio"


class ModelVariant(str, Enum):
    """Model capability variants."""

    DEFAULT = "default"
    REASONING = "reasoning"
    FAST = "fast"


@dataclass
class ModelMapping:
    """Maps model variants to actual model names."""

    default: str
    reasoning: Optional[str] = None
    fast: Optional[str] = None

    def get_model(self, variant: str) -> str:
        """Get model name for a variant, falling back to default if not specified."""
        if variant == ModelVariant.REASONING and self.reasoning:
            return self.reasoning
        elif variant == ModelVariant.FAST and self.fast:
            return self.fast
        else:
            return self.default

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "ModelMapping":
        return cls(**data)


@dataclass
class ProviderProfile:
    """Configuration for a specific LLM provider."""

    provider: ProviderType
    models: ModelMapping
    api_key_env: Optional[str] = None
    endpoint_env: Optional[str] = None
    api_version: Optional[str] = None
    base_url: Optional[str] = None
    region: Optional[str] = None  # For AWS Bedrock

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProviderProfile":
        """Create ProviderProfile from dictionary."""
        data = data.copy()

        # Convert provider string to enum
        if isinstance(data.get("provider"), str):
            data["provider"] = ProviderType(data["provider"])

        # Convert models dict to ModelMapping
        if isinstance(data.get("models"), dict):
            data["models"] = ModelMapping.from_dict(data["models"])

        return cls(**data)

    def get_runtime_config(self) -> Dict[str, Any]:
        """Get runtime configuration with resolved environment variables."""
        config = {"provider": self.provider.value}

        # Resolve environment variables
        if self.api_key_env:
            api_key = os.getenv(self.api_key_env)
            if not api_key and self.provider != ProviderType.AZURE_OPENAI:  # AZURE OPENAI can use integrated auth
                raise ValueError(f"Environment variable {self.api_key_env} not set")
            config["api_key"] = api_key  # type: ignore

        if self.endpoint_env:
            endpoint = os.getenv(self.endpoint_env)
            if endpoint:
                config["endpoint"] = endpoint

        # Add static configuration
        if self.api_version:
            config["api_version"] = self.api_version
        if self.base_url:
            config["base_url"] = self.base_url
        if self.region:
            config["region"] = self.region

        return config


@dataclass
class ModelParameters:
    """Model parameters for LLM calls."""

    temperature: float = 1.0
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelParameters":
        """Create ModelParameters from dictionary."""
        return cls(**data)

    def merge(self, other: Optional["ModelParameters"]) -> "ModelParameters":
        """Merge with another ModelParameters, with other taking precedence."""
        if not other:
            return self

        merged_dict = asdict(self)
        other_dict = asdict(other)

        # Update with non-None values from other
        for key, value in other_dict.items():
            if value is not None:
                merged_dict[key] = value

        return ModelParameters(**merged_dict)


@dataclass
class RoleConfig:
    """Configuration for a specific role (solver, judge, etc.)."""

    model_variant: str = ModelVariant.DEFAULT
    model_parameters: ModelParameters = field(default_factory=ModelParameters)
    persona: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RoleConfig":
        """Create RoleConfig from dictionary."""
        data = data.copy()

        # Convert model_parameters dict to ModelParameters
        if "model_parameters" in data and isinstance(data["model_parameters"], dict):
            data["model_parameters"] = ModelParameters.from_dict(data["model_parameters"])
        elif "model_parameters" not in data:
            data["model_parameters"] = ModelParameters()

        return cls(**data)

    def merge(self, override: Optional["RoleConfig"]) -> "RoleConfig":
        """Merge with an override config, with override taking precedence."""
        if not override:
            return self

        return RoleConfig(
            model_variant=override.model_variant or self.model_variant,
            model_parameters=self.model_parameters.merge(override.model_parameters),
            persona=override.persona or self.persona,
        )


@dataclass(slots=True)
class LLMConfig:
    """Top-level LLM configuration for an eval."""

    active_profile: str
    defaults: RoleConfig = field(default_factory=RoleConfig)
    overrides: Dict[str, RoleConfig] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMConfig":
        """Create LLMConfig from dictionary."""
        data = data.copy()

        # Convert defaults to RoleConfig
        if "defaults" in data and isinstance(data["defaults"], dict):
            data["defaults"] = RoleConfig.from_dict(data["defaults"])
        elif "defaults" not in data:
            data["defaults"] = RoleConfig()

        # Convert overrides to RoleConfig objects
        if "overrides" in data and isinstance(data["overrides"], dict):
            overrides = {}
            for role, config in data["overrides"].items():
                if isinstance(config, dict):
                    overrides[role] = RoleConfig.from_dict(config)
                else:
                    overrides[role] = config
            data["overrides"] = overrides

        return cls(**data)


@dataclass
class ResolvedLLMConfig:
    """Fully resolved configuration for a specific role."""

    provider: str
    model: str
    model_parameters: Dict[str, Any]
    persona: Optional[str] = None
    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    api_version: Optional[str] = None
    base_url: Optional[str] = None
    region: Optional[str] = None


class LLMConfigLoader:
    """Handles loading and merging eval configurations with provider profiles."""

    def __init__(self, profiles_path: Optional[str] = None):
        """Initialize the config loader."""
        self.profiles = self._load_profiles(profiles_path)

    def _load_profiles(self, profiles_path: Optional[str] = None) -> Dict[str, ProviderProfile]:
        """Load provider profiles from file."""
        # Determine path to profiles
        if profiles_path:
            path = Path(profiles_path)
        else:
            # Check environment variable
            env_path = os.getenv("EVAL_PROFILES_PATH")
            if env_path:
                path = Path(env_path)
            # Check local directory
            elif Path("./provider_profiles.json").exists():
                path = Path("./provider_profiles.json")
            # Check user config directory
            else:
                config_path = Path.home() / ".config" / "eval" / "provider_profiles.json"
                if config_path.exists():
                    path = config_path
                else:
                    raise FileNotFoundError(
                        "No provider profiles found. Please create provider_profiles.json "
                        "or set EVAL_PROFILES_PATH environment variable."
                    )

        with open(path, "r") as f:
            data = json.load(f)

        # Convert to ProviderProfile objects
        profiles = {}
        for name, profile_data in data.get("profiles", {}).items():
            profiles[name] = ProviderProfile.from_dict(profile_data)

        return profiles

    def get_llm_config(self, llm_config: LLMConfig, role: str) -> ResolvedLLMConfig:
        # Get active profile name (can be overridden by environment)
        profile_name = os.getenv("EVAL_PROVIDER_PROFILE", llm_config.active_profile)

        if profile_name not in self.profiles:
            available = ", ".join(self.profiles.keys())
            raise ValueError(f"Profile '{profile_name}' not found. Available profiles: {available}")

        profile = self.profiles[profile_name]

        # Start with defaults and merge with role-specific overrides
        role_config = llm_config.defaults

        # Check for exact role match first
        if role in llm_config.overrides:
            role_config = role_config.merge(llm_config.overrides[role])
        else:
            # Check for dotted role variants (e.g., solver.reasoning)
            for override_role, override_config in llm_config.overrides.items():
                if override_role.startswith(f"{role}."):
                    role_config = role_config.merge(override_config)
                    break

        # Get actual model name from variant
        model_name = profile.models.get_model(role_config.model_variant)

        # Get runtime configuration from profile
        runtime_config = profile.get_runtime_config()

        # Build resolved configuration
        return ResolvedLLMConfig(
            provider=runtime_config["provider"],
            model=model_name,
            model_parameters=asdict(role_config.model_parameters),
            persona=role_config.persona,
            api_key=runtime_config.get("api_key"),
            endpoint=runtime_config.get("endpoint"),
            api_version=runtime_config.get("api_version"),
            base_url=runtime_config.get("base_url"),
            region=runtime_config.get("region"),
        )
