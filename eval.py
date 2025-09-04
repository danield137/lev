import argparse
import asyncio
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

from lev.dataset_loader import load_eval_with_mcps
from lev.runner import run_evals

EVAL_FILES_EXTENSION = ".evl"


async def run_mcp_evaluations(dataset_file: str, limit: Optional[int] = None):
    eval_files: list[str] = []
    
    if not dataset_file:
        # load all local files
        eval_files = [str(p) for p in Path(".").glob(f"*{EVAL_FILES_EXTENSION}")]
        print(f"Found {len(eval_files)} evaluation files in the current directory")
    else:
        # assume file is json, so if no extension is provided, add .json
        if not dataset_file.endswith(".evl"):
            dataset_file += ".evl"

        dataset_path = Path(dataset_file)
        if not dataset_path.exists():
            print(f"Error: Dataset file '{dataset_file}' not found")
            exit(1)

        eval_files.append(dataset_file)

    for eval_file in eval_files:
        print(f"Loading dataset from {eval_file}")
        resolved = load_eval_with_mcps(eval_file)

        # Run evaluations using shared infrastructure
        await run_evals(resolved.name, resolved.evals, resolved.provider_registry, resolved.mcp_registry, limit=limit)


async def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Run MCP evaluations from a dataset file")
    parser.add_argument(
        "dataset",
        nargs="?",
        default="fs_mcp_dataset.json",
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
