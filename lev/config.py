from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class EvalSolverOptions:
    max_reasoning_steps: int = 3
    max_retrospective_turns: int = 1


@dataclass(slots=True)
class EvalAskerOptions:
    max_turns: int = 1


@dataclass(slots=True)
class EvalExecution:
    mcps: list[str]
    solver: EvalSolverOptions | None = None
    asker: EvalAskerOptions | None = None


@dataclass(slots=True)
class ScorerConfig:
    type: str
    weight: float = 1.0
    mode: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Eval:
    id: str
    question: str
    execution: EvalExecution
    scoring: list[ScorerConfig]
    expectations: dict[str, Any] | None = None


@dataclass(slots=True)
class ModelConfig:
    provider: str
    model: str
    model_parameters: dict[str, Any]
    persona: str | None = None  # Persona key or direct system prompt


@dataclass(slots=True)
class RolesConfig:
    solver: ModelConfig | None = None
    asker: ModelConfig | None = None
    judge: ModelConfig | None = None
