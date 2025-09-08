from typing import Callable

from lev.config import ScorerConfig
from lev.judge import Judge
from lev.scoring import Scorer
from lev.scoring.contains_string import create_contains_string_scorer
from lev.scoring.llm_judge import create_llm_judge_scorer

# Registry of scorer factories
SCORER_FACTORIES: dict[str, Callable[..., Scorer]] = {
    "llm_judge": create_llm_judge_scorer,
    "contains_string": create_contains_string_scorer,
}


def build_scorers(scoring_config: list[ScorerConfig], judge: Judge) -> list[tuple[float, Scorer]]:
    """
    Build a list of weighted scorers from configuration.

    Args:
        scoring_config: List of scorer configuration dictionaries
        judge: Judge instance for LLM-based scoring

    Returns:
        List of (weight, scorer) tuples
    """
    weighted_scorers = []

    for config in scoring_config:
        if isinstance(config, ScorerConfig):
            if config.type not in SCORER_FACTORIES:
                raise ValueError(f"Unknown scorer type: {config.type}")

            weight = config.weight

            # Build kwargs from all config fields
            kwargs = {"judge": judge, **config.parameters}
            if config.mode is not None:
                kwargs["mode"] = config.mode

            # Call the factory method
            factory = SCORER_FACTORIES[config.type]
            scorer = factory(**kwargs)
        else:
            raise ValueError(f"Invalid scorer configuration: {config}")

        weighted_scorers.append((weight, scorer))

    return weighted_scorers
