"""Summary metrics for accuracy and macro-F1 degradation across SNR."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RobustnessAnalysis:
    """Per-condition degradation values and one summary per trained model."""

    condition_metrics: pd.DataFrame
    summary: pd.DataFrame


def calculate_robustness_metrics(
    results: pd.DataFrame,
    *,
    group_columns: Sequence[str] = ("architecture", "training_condition"),
) -> RobustnessAnalysis:
    """Calculate drops, retention, slopes, and normalized SNR-curve areas.

    Clean rows use a null snr_db. Accuracy drop is clean minus condition accuracy;
    retention is condition divided by clean. Slopes use ordinary least squares on
    finite SNR points. Normalized area is trapezoidal area divided by the observed
    SNR range, so it has the same 0-1 scale as accuracy or macro F1.
    """
    required = {
        *group_columns,
        "condition",
        "snr_db",
        "accuracy",
        "macro_f1",
    }
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Robustness results are missing columns: {sorted(missing)}")

    condition_tables: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    grouped = results.groupby(list(group_columns), dropna=False, sort=False)
    for group_key, group in grouped:
        group_values = (
            group_key if isinstance(group_key, tuple) else (group_key,)
        )
        clean_rows = group.loc[group["snr_db"].isna()]
        if len(clean_rows) != 1:
            raise ValueError(
                f"Each robustness group requires exactly one clean row; "
                f"found {len(clean_rows)} for {group_values}."
            )
        clean = clean_rows.iloc[0]
        enriched = group.copy()
        enriched["accuracy_drop"] = float(clean["accuracy"]) - enriched["accuracy"]
        enriched["macro_f1_drop"] = float(clean["macro_f1"]) - enriched["macro_f1"]
        enriched["accuracy_retention"] = (
            enriched["accuracy"] / float(clean["accuracy"])
            if float(clean["accuracy"]) != 0
            else np.nan
        )
        enriched["macro_f1_retention"] = (
            enriched["macro_f1"] / float(clean["macro_f1"])
            if float(clean["macro_f1"]) != 0
            else np.nan
        )
        condition_tables.append(enriched)

        noisy = enriched.loc[enriched["snr_db"].notna()].sort_values("snr_db")
        snr_values = noisy["snr_db"].to_numpy(dtype=float)
        accuracy_values = noisy["accuracy"].to_numpy(dtype=float)
        macro_f1_values = noisy["macro_f1"].to_numpy(dtype=float)
        accuracy_slope = np.nan
        macro_f1_slope = np.nan
        accuracy_area = np.nan
        macro_f1_area = np.nan
        if len(noisy) >= 2 and np.ptp(snr_values) > 0:
            accuracy_slope = float(np.polyfit(snr_values, accuracy_values, 1)[0])
            macro_f1_slope = float(np.polyfit(snr_values, macro_f1_values, 1)[0])
            snr_range = float(snr_values[-1] - snr_values[0])
            accuracy_area = float(np.trapezoid(accuracy_values, snr_values) / snr_range)
            macro_f1_area = float(np.trapezoid(macro_f1_values, snr_values) / snr_range)

        summary_row: dict[str, object] = {
            column: value for column, value in zip(group_columns, group_values)
        }
        summary_row.update(
            {
                "clean_accuracy": float(clean["accuracy"]),
                "clean_macro_f1": float(clean["macro_f1"]),
                "accuracy_slope_per_db": accuracy_slope,
                "macro_f1_slope_per_db": macro_f1_slope,
                "normalized_accuracy_snr_auc": accuracy_area,
                "normalized_macro_f1_snr_auc": macro_f1_area,
                "worst_noisy_accuracy": (
                    float(noisy["accuracy"].min()) if not noisy.empty else np.nan
                ),
                "worst_noisy_macro_f1": (
                    float(noisy["macro_f1"].min()) if not noisy.empty else np.nan
                ),
            }
        )
        summary_rows.append(summary_row)

    return RobustnessAnalysis(
        condition_metrics=pd.concat(condition_tables, ignore_index=True),
        summary=pd.DataFrame(summary_rows),
    )


def save_robustness_analysis(
    analysis: RobustnessAnalysis,
    output_directory: str | Path,
) -> dict[str, Path]:
    """Persist the master condition table and model-level robustness summary."""
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "condition_metrics": output / "condition_metrics.csv",
        "summary": output / "robustness_summary.csv",
        "definitions": output / "robustness_metric_definitions.json",
    }
    analysis.condition_metrics.to_csv(paths["condition_metrics"], index=False)
    analysis.summary.to_csv(paths["summary"], index=False)
    definitions = {
        "accuracy_drop": "clean_accuracy - condition_accuracy",
        "relative_performance_retention": "condition_metric / clean_metric",
        "slope_per_db": "ordinary least-squares slope over finite SNR points",
        "normalized_snr_auc": (
            "trapezoidal area over finite SNR points divided by observed SNR range"
        ),
    }
    paths["definitions"].write_text(
        json.dumps(definitions, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return paths
