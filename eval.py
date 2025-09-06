import argparse
import asyncio
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from termcolor import colored

load_dotenv()

from lev.loader import load_manifest
from lev.runner import run_evals

EVAL_MANIFEST_FILES_EXTENSION = ".evl"


async def run_mcp_evaluations(manifest_file: str, limit: Optional[int] = None):
    manifest_files: list[str] = []

    if not manifest_file:
        # load all local files
        manifest_files = [str(p) for p in Path(".").glob(f"*{EVAL_MANIFEST_FILES_EXTENSION}")]
        print(f"Found {len(manifest_files)} evaluation files in the current directory")
    else:
        # assume file is json, so if no extension is provided, add .json
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

        # Run evaluations using shared infrastructure
        await run_evals(
            resolved.name,
            resolved.evals,
            resolved.provider_registry,
            resolved.mcp_registry,
            limit=limit,
            result_sink=resolved.result_sink,
        )


async def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Run MCP evaluations from a dataset file")
    parser.add_argument(
        "dataset",
        nargs="?",
        help="Path to the dataset JSON file (default: fs_mcp_dataset.json)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit the number of evaluation scenarios to run (for testing)",
    )

    args = parser.parse_args()

    await run_mcp_evaluations(args.dataset, limit=args.limit)


if __name__ == "__main__":
    asyncio.run(main())
