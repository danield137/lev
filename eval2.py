import argparse
import asyncio
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from termcolor import colored

load_dotenv()

from lev.loader import load_manifest
from lev.runner2 import run_host_evals

EVAL_MANIFEST_FILES_EXTENSION = ".evl"


async def run_mcp_host_evaluations(manifest_file: str, limit: Optional[int] = None):
    manifest_files: list[str] = []

    if not manifest_file:
        # load all local files
        manifest_files = [str(p) for p in Path(".").glob(f"*{EVAL_MANIFEST_FILES_EXTENSION}")]
        print(f"Found {len(manifest_files)} evaluation files in the current directory")
    else:
        # assume file is .evl, so if no extension is provided, add .evl
        if not manifest_file.endswith(".evl"):
            manifest_file += ".evl"

        dataset_path = Path(manifest_file)
        if not dataset_path.exists():
            print(f"Error: Manifest file '{manifest_file}' not found")
            exit(1)

        manifest_files.append(manifest_file)

    for eval_file in manifest_files:
        print(f"Loading dataset from {colored(eval_file, 'yellow')}")
        resolved = load_manifest(eval_file)

        # Run evaluations using new host-centric infrastructure
        await run_host_evals(
            resolved.name,
            resolved.evals,
            resolved.provider_registry,
            resolved.mcp_registry,
            limit=limit,
        )


async def main():
    """Main evaluation function using McpHost."""
    parser = argparse.ArgumentParser(description="Run MCP Host evaluations from a dataset file")
    parser.add_argument(
        "dataset",
        nargs="?",
        help="Path to the dataset .evl file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit the number of evaluation scenarios to run (for testing)",
    )

    args = parser.parse_args()

    await run_mcp_host_evaluations(args.dataset, limit=args.limit)


if __name__ == "__main__":
    asyncio.run(main())
