import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from lev.core.chat_history import ChatHistory
from lev.results import McpEvaluationResult


class TsvResultSink:
    """CSV file sink for evaluation results."""

    def __init__(self, file_path: str):
        """Initialize CSV sink with target file path."""
        self.file_path = file_path

    def write(self, results: list[McpEvaluationResult]) -> None:
        """Write results to CSV file, creating header if file is new."""
        if not results:
            return

        path = Path(self.file_path)
        file_exists = path.exists() and path.stat().st_size > 0

        with open(self.file_path, "a", newline="", encoding="utf-8") as csvfile:
            # Get field names from the first result
            sample_dict = self._result_to_dict(results[0])
            fieldnames = list(sample_dict.keys())

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter="\t")

            # Write header if file is new
            if not file_exists:
                writer.writeheader()

            # Write all results
            for result in results:
                writer.writerow(self._result_to_dict(result))

    def _result_to_dict(self, result: McpEvaluationResult) -> dict[str, Any]:
        """Convert McpEvaluationResult to flat dictionary suitable for CSV."""
        data = asdict(result)

        # Turn ChatHistory → single string trace
        if isinstance(data.get("conversation"), ChatHistory):
            data["conversation"] = data["conversation"].render_trace()

        # JSON-encode remaining complex fields
        complex_fields = {"conversation", "mcps", "tool_calls_sequence", "individual_scores"}

        for field in complex_fields:
            if field in data and data[field] is not None:
                data[field] = json.dumps(data[field], ensure_ascii=False)

        return data


def create_tsv_result_sink(manifest_name: str) -> TsvResultSink:
    """Create a TSV result sink with timestamped filename."""
    epoch = int(time.time())
    filename = f"{manifest_name}_results_{epoch}.tsv"
    return TsvResultSink(filename)
