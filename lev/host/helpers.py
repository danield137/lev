import csv
import datetime
import io
import json
import logging
from typing import Any

_mcp_logger = logging.getLogger("telemetry.mcp.calls")


def _csv_row(values: list[str]) -> str:
    """Convert a list of values to a CSV row string."""
    buf = io.StringIO()
    csv.writer(buf).writerow(values)
    return buf.getvalue().strip()


def _approx_tokens(text: str) -> int:
    """Approximate token count using simple word splitting."""
    return len(text.split())


def log_mcp_call(server_name: str, tool_name: str, arguments: dict[str, Any], response: dict[str, Any]):
    """Log an MCP call for tracing."""
    if _mcp_logger.isEnabledFor(logging.INFO):
        resp_text = json.dumps(response, ensure_ascii=False)
        _mcp_logger.info(
            _csv_row(
                [
                    datetime.datetime.now(tz=datetime.timezone.utc).isoformat(timespec="milliseconds") + "Z",
                    server_name,
                    tool_name,
                    json.dumps(arguments, ensure_ascii=False),
                    str(_approx_tokens(resp_text)),
                    str(len(resp_text.encode())),
                ]
            )
        )
