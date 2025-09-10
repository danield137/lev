from typing import Callable

from lev.config import ScorerConfig
from lev.context_compressor import ContextCompressor
from lev.judge import Judge
from lev.scoring import Scorer
from lev.scoring.contains_string import create_contains_string_scorer
from lev.scoring.llm import create_llm_critique_scorer, create_llm_extract_value_scorer
from lev.scoring.deterministic import (
    create_tool_call_count_scorer,
    create_tool_call_input_scorer,
    create_tool_call_output_scorer,
)

def create_llm_critique_scorer_wrapper(judge: Judge, **kwargs) -> Scorer:
    """Wrapper to create LLMCritiqueScorer with judge dependencies."""
    context_compressor = ContextCompressor(judge.llm_provider)
    return create_llm_critique_scorer(judge.llm_provider, context_compressor, judge.system_prompt, **kwargs)

def create_llm_extract_value_scorer_wrapper(judge: Judge, **kwargs) -> Scorer:
    """Wrapper to create LLMExtractValueScorer with judge dependencies."""
    expected = kwargs.pop("expected", None)
    if expected is None:
        raise ValueError("Missing required 'expected' parameter for LLMExtractValueScorer.")
    return create_llm_extract_value_scorer(judge.llm_provider, judge.system_prompt, expected, **kwargs)

# Registry of scorer factories
SCORER_FACTORIES: dict[str, Callable[..., Scorer]] = {
    "llm_critique": create_llm_critique_scorer_wrapper,
    "llm_extract": create_llm_extract_value_scorer_wrapper,
    "contains_string": create_contains_string_scorer,
    "tool_call_count": create_tool_call_count_scorer,
    "tool_call_input": create_tool_call_input_scorer,
    "tool_call_output": create_tool_call_output_scorer,
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
