"""Classification metrics, model prediction collection, and result storage."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch import Tensor, nn


@dataclass(frozen=True)
class ClassificationResult:
    """Complete single-label classification evaluation output."""

    summary: dict[str, float]
    per_class: pd.DataFrame
    confusion_matrix: np.ndarray
    targets: np.ndarray
    predictions: np.ndarray


def calculate_classification_metrics(
    targets: Sequence[int] | np.ndarray,
    predictions: Sequence[int] | np.ndarray,
    class_names: Sequence[str],
) -> ClassificationResult:
    """Calculate overall, macro, per-class, and confusion-matrix metrics."""
    target_values = np.asarray(targets, dtype=np.int64)
    prediction_values = np.asarray(predictions, dtype=np.int64)
    if target_values.ndim != 1 or prediction_values.ndim != 1:
        raise ValueError("targets and predictions must be one-dimensional.")
    if len(target_values) == 0 or len(target_values) != len(prediction_values):
        raise ValueError("targets and predictions must have equal non-zero length.")
    if not class_names:
        raise ValueError("class_names must not be empty.")

    labels = np.arange(len(class_names), dtype=np.int64)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        target_values,
        prediction_values,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        target_values,
        prediction_values,
        labels=labels,
        average=None,
        zero_division=0,
    )
    summary = {
        "accuracy": float(accuracy_score(target_values, prediction_values)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "num_samples": float(len(target_values)),
    }
    per_class = pd.DataFrame(
        {
            "label": labels,
            "class_name": list(class_names),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support.astype(int),
        }
    )
    matrix = confusion_matrix(
        target_values,
        prediction_values,
        labels=labels,
    )
    return ClassificationResult(
        summary=summary,
        per_class=per_class,
        confusion_matrix=matrix,
        targets=target_values,
        predictions=prediction_values,
    )


def collect_model_predictions(
    model: nn.Module,
    data_loader: Iterable[Mapping[str, object]],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run deterministic inference and collect targets and predicted labels."""
    model.eval()
    targets: list[int] = []
    predictions: list[int] = []
    with torch.inference_mode():
        for batch in data_loader:
            features = batch["features"]
            labels = batch["label"]
            if not isinstance(features, Tensor) or not isinstance(labels, Tensor):
                raise TypeError("DataLoader batches require tensor features and label.")
            logits = model(features.to(device))
            batch_predictions = logits.argmax(dim=1).cpu()
            targets.extend(labels.cpu().tolist())
            predictions.extend(batch_predictions.tolist())
    return (
        np.asarray(targets, dtype=np.int64),
        np.asarray(predictions, dtype=np.int64),
    )


def save_classification_result(
    result: ClassificationResult,
    output_directory: str | Path,
    *,
    class_names: Sequence[str],
    sample_ids: Sequence[str] | None = None,
    prediction_metadata: Mapping[str, Sequence[object]] | None = None,
) -> dict[str, Path]:
    """Save summary, per-class metrics, confusion matrix, and predictions."""
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output / "summary.json",
        "per_class": output / "per_class_metrics.csv",
        "confusion_matrix": output / "confusion_matrix.csv",
        "predictions": output / "predictions.csv",
    }
    paths["summary"].write_text(
        json.dumps(result.summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result.per_class.to_csv(paths["per_class"], index=False)
    matrix_table = pd.DataFrame(
        result.confusion_matrix,
        index=class_names,
        columns=class_names,
    )
    matrix_table.index.name = "actual"
    matrix_table.to_csv(paths["confusion_matrix"])
    if sample_ids is not None and len(sample_ids) != len(result.targets):
        raise ValueError("sample_ids length must match evaluated targets.")
    prediction_table = pd.DataFrame(
        {
            "sample_id": (
                list(sample_ids)
                if sample_ids is not None
                else [None] * len(result.targets)
            ),
            "target": result.targets,
            "prediction": result.predictions,
        }
    )
    if prediction_metadata is not None:
        reserved_columns = set(prediction_table.columns)
        for column_name, values in prediction_metadata.items():
            if column_name in reserved_columns:
                raise ValueError(
                    f"Prediction metadata cannot replace '{column_name}'."
                )
            if len(values) != len(result.targets):
                raise ValueError(
                    f"Prediction metadata '{column_name}' length must match "
                    "evaluated targets."
                )
            prediction_table[column_name] = list(values)
    prediction_table.to_csv(paths["predictions"], index=False)
    return paths
