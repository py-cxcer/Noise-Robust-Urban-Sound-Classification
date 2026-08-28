"""Tests for classification and robustness evaluation outputs."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from urban_sound_robustness.evaluation import (
    calculate_classification_metrics,
    calculate_robustness_metrics,
    collect_model_predictions,
    save_classification_result,
    save_robustness_analysis,
)


def test_classification_metrics_include_macro_and_per_class_values() -> None:
    """Known predictions should produce complete structured metrics."""
    result = calculate_classification_metrics(
        targets=[0, 0, 1, 1],
        predictions=[0, 1, 1, 1],
        class_names=["zero", "one"],
    )

    assert result.summary["accuracy"] == pytest.approx(0.75)
    assert result.summary["macro_precision"] == pytest.approx((1.0 + 2 / 3) / 2)
    assert list(result.per_class["class_name"]) == ["zero", "one"]
    assert result.confusion_matrix.tolist() == [[1, 1], [0, 2]]


def test_classification_result_storage_writes_all_tables(tmp_path: Path) -> None:
    """Metrics should be available as JSON and readable CSV files."""
    result = calculate_classification_metrics(
        [0, 1],
        [0, 1],
        ["zero", "one"],
    )

    paths = save_classification_result(
        result,
        tmp_path,
        class_names=["zero", "one"],
        sample_ids=["a.wav", "b.wav"],
        prediction_metadata={
            "condition": ["clean", "clean"],
            "noise_path": ["", ""],
        },
    )

    assert set(paths) == {
        "summary",
        "per_class",
        "confusion_matrix",
        "predictions",
    }
    assert all(path.is_file() for path in paths.values())
    assert pd.read_csv(paths["predictions"])["sample_id"].tolist() == [
        "a.wav",
        "b.wav",
    ]
    prediction_table = pd.read_csv(paths["predictions"], keep_default_na=False)
    assert prediction_table["condition"].tolist() == ["clean", "clean"]
    assert prediction_table["noise_path"].tolist() == ["", ""]


def test_classification_result_rejects_reserved_prediction_metadata(
    tmp_path: Path,
) -> None:
    """Corruption provenance cannot overwrite core prediction columns."""
    result = calculate_classification_metrics([0, 1], [0, 1], ["zero", "one"])

    with pytest.raises(ValueError, match="cannot replace"):
        save_classification_result(
            result,
            tmp_path,
            class_names=["zero", "one"],
            prediction_metadata={"target": [99, 99]},
        )


def test_collect_model_predictions_uses_common_batch_structure() -> None:
    """Inference should consume structured features/label dictionaries."""
    samples = [
        {"features": torch.tensor([1.0, 0.0]), "label": torch.tensor(0)},
        {"features": torch.tensor([0.0, 1.0]), "label": torch.tensor(1)},
    ]
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2))

    targets, predictions = collect_model_predictions(
        model,
        DataLoader(samples, batch_size=2),
        torch.device("cpu"),
    )

    assert targets.tolist() == [0, 1]
    assert predictions.tolist() == [0, 1]


def _robustness_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "architecture": "cnn",
                "training_condition": "baseline",
                "condition": "clean",
                "snr_db": np.nan,
                "accuracy": 0.9,
                "macro_f1": 0.8,
            },
            {
                "architecture": "cnn",
                "training_condition": "baseline",
                "condition": "snr_20db",
                "snr_db": 20.0,
                "accuracy": 0.8,
                "macro_f1": 0.7,
            },
            {
                "architecture": "cnn",
                "training_condition": "baseline",
                "condition": "snr_10db",
                "snr_db": 10.0,
                "accuracy": 0.6,
                "macro_f1": 0.5,
            },
            {
                "architecture": "cnn",
                "training_condition": "baseline",
                "condition": "snr_0db",
                "snr_db": 0.0,
                "accuracy": 0.4,
                "macro_f1": 0.3,
            },
        ]
    )


def test_robustness_metrics_calculate_drops_retention_slope_and_auc() -> None:
    """Robustness summaries should follow their documented definitions."""
    analysis = calculate_robustness_metrics(_robustness_fixture())
    zero_db = analysis.condition_metrics.loc[
        analysis.condition_metrics["condition"] == "snr_0db"
    ].iloc[0]
    summary = analysis.summary.iloc[0]

    assert zero_db["accuracy_drop"] == pytest.approx(0.5)
    assert zero_db["accuracy_retention"] == pytest.approx(0.4 / 0.9)
    assert summary["accuracy_slope_per_db"] == pytest.approx(0.02)
    assert summary["normalized_accuracy_snr_auc"] == pytest.approx(0.6)


def test_robustness_analysis_requires_one_clean_row() -> None:
    """Drops cannot be defined without an unambiguous clean reference."""
    fixture = _robustness_fixture()
    without_clean = fixture.loc[fixture["condition"] != "clean"]

    with pytest.raises(ValueError, match="exactly one clean"):
        calculate_robustness_metrics(without_clean)


def test_robustness_storage_writes_definitions(tmp_path: Path) -> None:
    """Saved analysis should include exact mathematical definitions."""
    analysis = calculate_robustness_metrics(_robustness_fixture())

    paths = save_robustness_analysis(analysis, tmp_path)

    assert all(path.is_file() for path in paths.values())
    assert "normalized_snr_auc" in paths["definitions"].read_text(encoding="utf-8")
