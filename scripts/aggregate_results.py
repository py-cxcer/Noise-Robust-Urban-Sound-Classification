"""Validate and aggregate the six final robustness evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from urban_sound_robustness.evaluation import (
    aggregate_evaluation_results,
    save_aggregated_results,
)
from urban_sound_robustness.utils import configure_logging, find_project_root


def parse_arguments() -> argparse.Namespace:
    """Parse optional input/output overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-directory",
        type=Path,
        default=None,
        help="Override results/robustness.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
        help="Override results/analysis/final_robustness.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing aggregate files after inspection.",
    )
    return parser.parse_args()


def main() -> int:
    """Aggregate validated final results and print the key summary."""
    arguments = parse_arguments()
    started_at = time.perf_counter()
    project_root = find_project_root(Path(__file__))
    input_directory = (
        project_root / "results" / "robustness"
        if arguments.input_directory is None
        else arguments.input_directory.expanduser().resolve()
    )
    output_directory = (
        project_root / "results" / "analysis" / "final_robustness"
        if arguments.output_directory is None
        else arguments.output_directory.expanduser().resolve()
    )
    aggregated = aggregate_evaluation_results(input_directory)
    if (
        output_directory.is_dir()
        and any(output_directory.iterdir())
        and not arguments.overwrite
    ):
        raise FileExistsError(
            f"Aggregate output already exists and is not empty: "
            f"{output_directory}. Inspect it, then pass --overwrite only when "
            "replacement is intended."
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(output_directory / "aggregation.log")
    logger.info("Reading final evaluations from: %s", input_directory)
    paths = save_aggregated_results(aggregated, output_directory)
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    summary["elapsed_seconds"] = time.perf_counter() - started_at
    logger.info(
        "Aggregated %d models and %d condition results",
        summary["num_models"],
        summary["num_condition_results"],
    )
    logger.info("Final analysis summary: %s", paths["summary"])
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
