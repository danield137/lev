from __future__ import annotations

from collections.abc import Mapping
from lev.scoring import Score, Scorer, ScoringContext

class ToolCallCountScorer(Scorer):
    """
    Validate call-count constraints for each tool.

    Parameters
    ----------
    calls : Mapping[str, Mapping[str, int]]
        tool → {exact|min|max}.  Only one of exact / min / max may be present.
    order_matters : bool, default False
        When true, also require the sequence of calls to follow the key order.
    """

    def __init__(
        self,
        calls: Mapping[str, Mapping[str, int]],
        *,
        order_matters: bool = False,
        **kwargs,
    ) -> None:
        self.calls = calls
        self.order_matters = order_matters

    @property
    def display_name(self) -> str:
        return "tool_call_count"

    async def score(self, ctx: ScoringContext) -> Score:
        if not ctx.tool_calls:
            # If no tool calls were made, check if any were required
            for tool, spec in self.calls.items():
                min_required = spec.get("min", 0)
                exact_required = spec.get("exact")
                if exact_required and exact_required > 0:
                    return Score(0.0, f"{tool}: expected {exact_required}, got 0")
                if min_required > 0:
                    return Score(0.0, f"{tool}: min {min_required}, got 0")
            return Score(1.0, "no tool calls required or made")

        # Count tool calls by name
        call_hist: dict[str, int] = {}
        call_sequence: list[str] = []
        
        for tool_call in ctx.tool_calls:
            tool_name = tool_call.get("tool_name", "unknown")
            call_hist[tool_name] = call_hist.get(tool_name, 0) + 1
            call_sequence.append(tool_name)

        # Validate count constraints
        for tool, spec in self.calls.items():
            actual_count = call_hist.get(tool, 0)
            
            if "exact" in spec:
                expected = spec["exact"]
                if actual_count != expected:
                    return Score(0.0, f"{tool}: expected exactly {expected}, got {actual_count}")
            else:
                if "min" in spec and actual_count < spec["min"]:
                    return Score(0.0, f"{tool}: min {spec['min']}, got {actual_count}")
                if "max" in spec and actual_count > spec["max"]:
                    return Score(0.0, f"{tool}: max {spec['max']}, got {actual_count}")

        # Validate sequence order if required
        if self.order_matters:
            expected_tools = list(self.calls.keys())
            filtered_sequence = [tool for tool in call_sequence if tool in expected_tools]
            
            # Check if sequence starts with expected order
            for i, expected_tool in enumerate(expected_tools):
                if i >= len(filtered_sequence) or filtered_sequence[i] != expected_tool:
                    return Score(0.0, f"sequence mismatch: expected {expected_tools}, got {filtered_sequence}")

        return Score(1.0, f"call counts satisfied: {call_hist}")


def create_tool_call_count_scorer(calls: Mapping[str, Mapping[str, int]], order_matters: bool = False, **kwargs) -> ToolCallCountScorer:
    """Factory method to create a tool call count scorer."""
    return ToolCallCountScorer(calls=calls, order_matters=order_matters)
