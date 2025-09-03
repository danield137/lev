import os
import pathlib
import sys
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP

# Create MCP server
mcp = FastMCP("Filesystem MCP Server")


@mcp.tool()
def list_tree(path: str = ".", depth: int = 2) -> List[str]:
    """
    List files and directories recursively up to a specified depth.

    Args:
        path: Directory path to list (default: current directory)
        depth: Maximum depth to recurse (default: 2)

    Returns:
        List of relative file paths
    """
    if not isinstance(path, str):
        raise ValueError("Path must be a string")
    if not isinstance(depth, int) or depth < 0:
        raise ValueError("Depth must be a non-negative integer")

    # Convert to absolute path and validate
    abs_path = pathlib.Path(path).resolve()
    if not abs_path.exists():
        raise FileNotFoundError(f"Path '{path}' does not exist")
    if not abs_path.is_dir():
        raise ValueError(f"Path '{path}' is not a directory")

    files = []

    def _collect_files(current_path: pathlib.Path, current_depth: int, max_depth: int):
        if current_depth > max_depth:
            return

        try:
            for item in sorted(current_path.iterdir()):
                # Get relative path from the original requested path
                rel_path = item.relative_to(abs_path)
                files.append(str(rel_path))

                # Recurse into subdirectories
                if item.is_dir() and current_depth < max_depth:
                    _collect_files(item, current_depth + 1, max_depth)
        except PermissionError:
            # Skip directories we can't read
            pass

    _collect_files(abs_path, 0, depth)

    return files


@mcp.tool()
def read_file(path: str, max_bytes: int = 32768) -> Dict[str, Any]:
    """
    Read the contents of a text file.

    Args:
        path: File path to read
        max_bytes: Maximum bytes to read (default: 32768)

    Returns:
        Dictionary with content, truncated flag, and bytes_read
    """
    if not path:
        raise ValueError("Path is required")
    if not isinstance(path, str):
        raise ValueError("Path must be a string")
    if not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")

    # Convert to absolute path and validate
    abs_path = pathlib.Path(path).resolve()
    if not abs_path.exists():
        raise FileNotFoundError(f"File '{path}' does not exist")
    if not abs_path.is_file():
        raise ValueError(f"Path '{path}' is not a file")

    # Read file content
    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            content = f.read(max_bytes)

        # Check if file was truncated
        truncated = len(content) == max_bytes
        if truncated:
            # Check if there's more content
            with open(abs_path, "r", encoding="utf-8") as f:
                f.seek(max_bytes)
                next_char = f.read(1)
                truncated = bool(next_char)

        return {"content": content, "truncated": truncated, "bytes_read": len(content.encode("utf-8"))}

    except UnicodeDecodeError:
        raise ValueError(f"File '{path}' is not a valid text file (encoding error)")
    except PermissionError:
        raise PermissionError(f"Permission denied reading file '{path}'")


def main():
    """Run the filesystem MCP server."""
    # Check if output should be suppressed
    suppress_output = os.getenv("MCP_SUPPRESS_OUTPUT", "").lower() in ("1", "true", "yes")

    print("Starting Filesystem MCP Server")
    print("Available tools: list_tree, read_file")

    # Redirect both stdout and stderr to devnull if suppressing output
    if suppress_output:
        # Save original streams
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        # Redirect to devnull
        devnull = open(os.devnull, "w")
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            mcp.run(show_banner=False)
        finally:
            # Restore original streams
            devnull.close()
            sys.stdout = original_stdout
            sys.stderr = original_stderr
    else:
        mcp.run(show_banner=False)


if __name__ == "__main__":
    main()
