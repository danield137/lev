from __future__ import annotations

import re
from collections.abc import Mapping
from lev.scoring import Score, Scorer, ScoringContext

class ToolCallInputScorer(Scorer):
    """
    Validate that specific input fields match expected patterns.

    Parameters
    ----------
    inputs : Mapping[str, list[dict]]
        tool → list of {field, value, mode}
          mode ∈ {"exact", "contains", "regex"}
    """

    def __init__(self, inputs: Mapping[str, list[Mapping]], **kwargs) -> None:
        self.inputs = inputs

    @property
    def display_name(self) -> str:
        return "tool_call_input"

    async def score(self, ctx: ScoringContext) -> Score:
        if not ctx.tool_calls:
            if self.inputs:
                return Score(0.0, "no tool calls made but input validation expected")
            return Score(1.0, "no tool calls or input validation required")

        # Group tool calls by tool name
        tools_called: dict[str, list[dict]] = {}
        for tool_call in ctx.tool_calls:
            tool_name = tool_call.get("tool_name", "unknown")
            if tool_name not in tools_called:
                tools_called[tool_name] = []
            tools_called[tool_name].append(tool_call)

        # Validate inputs for each expected tool
        for tool, checks in self.inputs.items():
            if tool not in tools_called:
                return Score(0.0, f"missing tool calls for {tool}")

            # Validate against the first call (or could validate all calls)
            tool_call = tools_called[tool][0]
            arguments = tool_call.get("arguments", {})

            for check in checks:
                field = check["field"]
                expected_value = check["value"]
                mode = check.get("mode", "exact")

                if field not in arguments:
                    return Score(0.0, f"{tool}.{field} missing in arguments")

                actual_value = str(arguments[field])

                if mode == "exact":
                    if actual_value != expected_value:
                        return Score(0.0, f"{tool}.{field}: expected '{expected_value}', got '{actual_value}'")
                elif mode == "contains":
                    if expected_value not in actual_value:
                        return Score(0.0, f"{tool}.{field}: '{expected_value}' not found in '{actual_value}'")
                elif mode == "regex":
                    if not re.search(expected_value, actual_value):
                        return Score(0.0, f"{tool}.{field}: pattern '{expected_value}' not matched in '{actual_value}'")
                else:
                    return Score(0.0, f"invalid mode '{mode}' for {tool}.{field}")

        return Score(1.0, "all input validations passed")

def create_tool_call_input_scorer(inputs: Mapping[str, list[Mapping]], **kwargs) -> ToolCallInputScorer:
    """Factory method to create a tool call input scorer."""
    return ToolCallInputScorer(inputs=inputs)
