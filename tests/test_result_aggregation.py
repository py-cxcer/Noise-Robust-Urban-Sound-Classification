"""Tests for final six-run result aggregation and plotting."""

import json
from pathlib import Path

import pandas as pd
import pytest

from urban_sound_robustness.evaluation import (
    ResultAggregationError,
    aggregate_evaluation_results,
    save_aggregated_results,
)


CONDITIONS = {
    "clean": None,
    "snr_20db": 20.0,
    "snr_10db": 10.0,
    "snr_0db": 0.0,
}
GROUPS = [
    (architecture, training_condition)
    for architecture in ("cnn", "crnn", "resnet18")
    for training_condition in ("baseline", "augmented")
]


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _metric_value(
    architecture: str,
    training_condition: str,
    condition_index: int,
) -> float:
    architecture_offset = {"cnn": 0.0, "crnn": 0.03, "resnet18": 0.06}[
        architecture
    ]
    augmentation_offset = 0.1 if training_condition == "augmented" else 0.0
    return 0.75 + architecture_offset + augmentation_offset - 0.1 * condition_index


def _write_fixture_run(
    root: Path,
    architecture: str,
    training_condition: str,
) -> Path:
    experiment_id = f"{architecture}_{training_condition}_run"
    run_directory = root / experiment_id / "test"
    summary_path = run_directory / "evaluation_summary.json"
    _write_json(
        summary_path,
        {
            "experiment_id": experiment_id,
            "architecture": architecture,
            "training_condition": training_condition,
            "split": "test",
            "smoke_run": False,
            "num_samples": 2,
            "conditions": {name: {} for name in CONDITIONS},
        },
    )
    _write_json(
        run_directory / "evaluation_protocol.json",
        {
            "split": "test",
            "folds": [10],
            "evaluation": {
                "conditions": [
                    {"name": name, "snr_db": snr}
                    for name, snr in CONDITIONS.items()
                ],
                "corruption_seed": 2025,
            },
            "resolved_noise_directory": str((root / "noise_test").resolve()),
            "noise_file_count": 1,
            "sample_limit": None,
        },
    )

    condition_rows = []
    clean_value = _metric_value(architecture, training_condition, 0)
    for condition_index, (condition, snr_db) in enumerate(CONDITIONS.items()):
        value = _metric_value(
            architecture,
            training_condition,
            condition_index,
        )
        condition_rows.append(
            {
                "experiment_id": experiment_id,
                "architecture": architecture,
                "training_condition": training_condition,
                "condition": condition,
                "snr_db": snr_db,
                "accuracy": value,
                "macro_precision": value,
                "macro_recall": value,
                "macro_f1": value,
                "num_samples": 2.0,
                "accuracy_drop": clean_value - value,
                "macro_f1_drop": clean_value - value,
                "accuracy_retention": value / clean_value,
                "macro_f1_retention": value / clean_value,
            }
        )
    pd.DataFrame(condition_rows).to_csv(
        run_directory / "condition_metrics.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "experiment_id": experiment_id,
                "architecture": architecture,
                "training_condition": training_condition,
                "clean_accuracy": clean_value,
                "clean_macro_f1": clean_value,
                "accuracy_slope_per_db": 0.01,
                "macro_f1_slope_per_db": 0.01,
                "normalized_accuracy_snr_auc": clean_value - 0.2,
                "normalized_macro_f1_snr_auc": clean_value - 0.2,
                "worst_noisy_accuracy": clean_value - 0.3,
                "worst_noisy_macro_f1": clean_value - 0.3,
            }
        ]
    ).to_csv(run_directory / "robustness_summary.csv", index=False)

    for condition_index, (condition, snr_db) in enumerate(CONDITIONS.items()):
        condition_directory = run_directory / "conditions" / condition
        condition_directory.mkdir(parents=True, exist_ok=True)
        value = _metric_value(
            architecture,
            training_condition,
            condition_index,
        )
        pd.DataFrame(
            [
                {
                    "label": 0,
                    "class_name": "zero",
                    "precision": value,
                    "recall": value,
                    "f1": value,
                    "support": 1,
                },
                {
                    "label": 1,
                    "class_name": "one",
                    "precision": value - 0.05,
                    "recall": value - 0.05,
                    "f1": value - 0.05,
                    "support": 1,
                },
            ]
        ).to_csv(condition_directory / "per_class_metrics.csv", index=False)
        is_clean = condition == "clean"
        pd.DataFrame(
            [
                {
                    "sample_id": "a.wav",
                    "target": 0,
                    "prediction": 0,
                    "class_name": "zero",
                    "fold": 10,
                    "condition": condition,
                    "target_snr_db": snr_db,
                    "achieved_snr_db": snr_db,
                    "noise_path": (
                        "" if is_clean else str(root / "noise_test" / "noise.wav")
                    ),
                    "noise_selection_seed": -1 if is_clean else 11,
                    "noise_applied": not is_clean,
                },
                {
                    "sample_id": "b.wav",
                    "target": 1,
                    "prediction": 1,
                    "class_name": "one",
                    "fold": 10,
                    "condition": condition,
                    "target_snr_db": snr_db,
                    "achieved_snr_db": snr_db,
                    "noise_path": (
                        "" if is_clean else str(root / "noise_test" / "noise.wav")
                    ),
                    "noise_selection_seed": -1 if is_clean else 22,
                    "noise_applied": not is_clean,
                },
            ]
        ).to_csv(condition_directory / "predictions.csv", index=False)
    return summary_path


def _write_complete_fixture(root: Path) -> list[Path]:
    return [
        _write_fixture_run(root, architecture, training_condition)
        for architecture, training_condition in GROUPS
    ]


def test_aggregation_builds_complete_tables_and_figures(tmp_path: Path) -> None:
    """Six compatible runs should produce every table and plot."""
    input_directory = tmp_path / "robustness"
    _write_complete_fixture(input_directory)

    aggregated = aggregate_evaluation_results(input_directory)

    assert len(aggregated.condition_metrics) == 24
    assert len(aggregated.robustness_summary) == 6
    assert len(aggregated.augmentation_effects) == 12
    assert len(aggregated.per_class_metrics) == 48
    cnn_clean = aggregated.augmentation_effects.loc[
        (aggregated.augmentation_effects["architecture"] == "cnn")
        & (aggregated.augmentation_effects["condition"] == "clean")
    ].iloc[0]
    assert cnn_clean["macro_f1_delta"] == pytest.approx(0.1)

    paths = save_aggregated_results(aggregated, tmp_path / "analysis")
    assert all(path.is_file() for path in paths.values())
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert summary["num_models"] == 6
    assert summary["num_condition_results"] == 24


def test_aggregation_rejects_changed_noise_assignment(tmp_path: Path) -> None:
    """Every model and SNR must use the identical selected noise segment."""
    input_directory = tmp_path / "robustness"
    summaries = _write_complete_fixture(input_directory)
    changed_predictions = (
        summaries[-1].parent
        / "conditions"
        / "snr_0db"
        / "predictions.csv"
    )
    table = pd.read_csv(changed_predictions)
    table.loc[0, "noise_selection_seed"] = 999
    table.to_csv(changed_predictions, index=False)

    with pytest.raises(ResultAggregationError, match="noise assignments"):
        aggregate_evaluation_results(input_directory)


def test_aggregation_requires_all_six_groups(tmp_path: Path) -> None:
    """A missing model/training pair must stop final analysis."""
    input_directory = tmp_path / "robustness"
    for architecture, training_condition in GROUPS[:-1]:
        _write_fixture_run(input_directory, architecture, training_condition)

    with pytest.raises(ResultAggregationError, match="missing"):
        aggregate_evaluation_results(input_directory)
