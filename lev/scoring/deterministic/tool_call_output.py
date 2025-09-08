from __future__ import annotations

import math
from collections.abc import Mapping
from lev.scoring import Score, Scorer, ScoringContext

def _deep_compare(
    expected: Mapping,
    actual: Mapping,
    *,
    tol: float,
    ignore_extra: bool,
) -> bool:
    """
    Deep comparison helper for nested dictionaries with numeric tolerance.
    """
    for k, v in expected.items():
        if k not in actual:
            return False
        av = actual[k]
        
        if isinstance(v, Mapping):
            if not isinstance(av, Mapping):
                return False
            if not _deep_compare(v, av, tol=tol, ignore_extra=ignore_extra):
                return False
        elif isinstance(v, list):
            if not isinstance(av, list) or len(v) != len(av):
                return False
            for i, item in enumerate(v):
                if isinstance(item, Mapping):
                    if not isinstance(av[i], Mapping):
                        return False
                    if not _deep_compare(item, av[i], tol=tol, ignore_extra=ignore_extra):
                        return False
                elif isinstance(item, (int, float)):
                    if not isinstance(av[i], (int, float)) or math.fabs(av[i] - item) > tol:
                        return False
                else:
                    if av[i] != item:
                        return False
        elif isinstance(v, (int, float)):
            if not isinstance(av, (int, float)) or math.fabs(av - v) > tol:
                return False
        else:
            if av != v:
                return False
    
    # Check for extra keys if not ignoring them
    if not ignore_extra and set(actual.keys()) - set(expected.keys()):
        return False
    
    return True

class ToolCallOutputScorer(Scorer):
    """
    Validate tool result payloads against expected JSON fragments.

    Parameters
    ----------
    results : Mapping[str, Mapping]
        tool → expected JSON fragment to match.
    tolerance : float, default 1e-6
        Numeric tolerance for floating point comparisons.
    ignore_extra : bool, default True
        Allow additional keys in actual result beyond expected ones.
    """

    def __init__(
        self,
        results: Mapping[str, Mapping],
        *,
        tolerance: float = 1e-6,
        ignore_extra: bool = True,
        **kwargs,
    ) -> None:
        self.results = results
        self.tolerance = tolerance
        self.ignore_extra = ignore_extra

    @property
    def display_name(self) -> str:
        return "tool_call_output"

    async def score(self, ctx: ScoringContext) -> Score:
        if not ctx.tool_calls:
            if self.results:
                return Score(0.0, "no tool calls made but output validation expected")
            return Score(1.0, "no tool calls or output validation required")

        # Group tool calls by tool name
        tools_called: dict[str, list[dict]] = {}
        for tool_call in ctx.tool_calls:
            tool_name = tool_call.get("tool_name", "unknown")
            if tool_name not in tools_called:
                tools_called[tool_name] = []
            tools_called[tool_name].append(tool_call)

        # Validate results for each expected tool
        for tool, expected_result in self.results.items():
            if tool not in tools_called:
                return Score(0.0, f"missing tool calls for {tool}")

            # Check the result of the first call (or could check all calls)
            tool_call = tools_called[tool][0]
            actual_result = tool_call.get("result", {})

            if not _deep_compare(
                expected_result,
                actual_result,
                tol=self.tolerance,
                ignore_extra=self.ignore_extra,
            ):
                return Score(0.0, f"result mismatch for {tool}: expected subset of {expected_result}, got {actual_result}")

        return Score(1.0, "all output validations passed")

def create_tool_call_output_scorer(
    results: Mapping[str, Mapping],
    tolerance: float = 1e-6,
    ignore_extra: bool = True,
    **kwargs
) -> ToolCallOutputScorer:
    """Factory method to create a tool call output scorer."""
    return ToolCallOutputScorer(results=results, tolerance=tolerance, ignore_extra=ignore_extra)
