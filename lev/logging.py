import logging
from pathlib import Path

from lev.manifest import EvalManifest


def configure_telemetry_logging(manifest: EvalManifest, suite_name: str) -> None:
    """
    Configure telemetry logging based on dataset configuration.

    Args:
        dataset: The full dataset configuration
        suite_name: Base name for the suite (used for log file naming)
    """
    # Check for logging configuration
    logging_config = manifest.logging or {}
    mcp_calls_enabled = logging_config.get("mcp_calls", False)

    # Get the MCP logger
    mcp_logger = logging.getLogger("telemetry.mcp.calls")

    if mcp_calls_enabled:
        # Set up file logging for MCP calls
        log_path = f"{suite_name}_mcp_log.csv"

        # Remove any existing handlers to avoid duplicates
        for handler in mcp_logger.handlers[:]:
            mcp_logger.removeHandler(handler)

        # Create file handler
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(message)s"))

        # Configure logger
        mcp_logger.setLevel(logging.INFO)
        mcp_logger.addHandler(file_handler)
        mcp_logger.propagate = False  # Prevent propagation to root logger

        # Write CSV header if file is new/empty
        try:
            if Path(log_path).stat().st_size == 0:
                mcp_logger.info("timestamp,server_name,tool_name,arguments,response_size_tokens,response_size_bytes")
        except (FileNotFoundError, OSError):
            # File doesn't exist yet, header will be written on first log
            mcp_logger.info("timestamp,server_name,tool_name,arguments,response_size_tokens,response_size_bytes")
    else:
        # Disable MCP call logging
        mcp_logger.setLevel(logging.CRITICAL)
        # Add null handler to avoid "No handler" warnings
        if not mcp_logger.handlers:
            mcp_logger.addHandler(logging.NullHandler())
        mcp_logger.propagate = False
