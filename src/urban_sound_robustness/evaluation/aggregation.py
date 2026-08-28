"""Aggregate and validate the six final robustness evaluations."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONDITION_ORDER = ("clean", "snr_20db", "snr_10db", "snr_0db")
CONDITION_LABELS = ("Clean", "20 dB", "10 dB", "0 dB")
CONDITION_SNR = {
    "clean": None,
    "snr_20db": 20.0,
    "snr_10db": 10.0,
    "snr_0db": 0.0,
}
EXPECTED_GROUPS = {
    (architecture, training_condition)
    for architecture in ("cnn", "crnn", "resnet18")
    for training_condition in ("baseline", "augmented")
}


class ResultAggregationError(ValueError):
    """Raised when final evaluations are missing or not directly comparable."""


@dataclass(frozen=True)
class AggregatedResults:
    """Validated master tables and the shared evaluation protocol."""

    condition_metrics: pd.DataFrame
    robustness_summary: pd.DataFrame
    robustness_effects: pd.DataFrame
    augmentation_effects: pd.DataFrame
    per_class_metrics: pd.DataFrame
    per_class_effects: pd.DataFrame
    protocol: dict[str, Any]


@dataclass(frozen=True)
class _RunArtifacts:
    """Loaded tables belonging to one checkpoint evaluation."""

    summary_path: Path
    summary: Mapping[str, Any]
    protocol: Mapping[str, Any]
    condition_metrics: pd.DataFrame
    robustness_summary: pd.DataFrame
    per_class_metrics: pd.DataFrame
    predictions: Mapping[str, pd.DataFrame]


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON mapping with a path-aware error."""
    if not path.is_file():
        raise ResultAggregationError(f"Required JSON file is missing: {path}")
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ResultAggregationError(f"Could not read JSON '{path}': {error}") from error
    if not isinstance(loaded, dict):
        raise ResultAggregationError(f"JSON root must be a mapping: {path}")
    return loaded


def _read_csv(path: Path) -> pd.DataFrame:
    """Read one non-empty CSV table with a path-aware error."""
    if not path.is_file():
        raise ResultAggregationError(f"Required CSV file is missing: {path}")
    try:
        table = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as error:
        raise ResultAggregationError(f"Could not read CSV '{path}': {error}") from error
    if table.empty:
        raise ResultAggregationError(f"Required CSV table is empty: {path}")
    return table


def _load_run(summary_path: Path) -> _RunArtifacts:
    """Load one complete test evaluation using paths relative to its summary."""
    run_directory = summary_path.parent
    summary = _read_json(summary_path)
    protocol = _read_json(run_directory / "evaluation_protocol.json")
    if summary.get("split") != "test" or bool(summary.get("smoke_run", True)):
        raise ResultAggregationError(
            f"Only non-smoke test evaluations can be aggregated: {summary_path}"
        )
    condition_names = set(summary.get("conditions", {}))
    if condition_names != set(CONDITION_ORDER):
        raise ResultAggregationError(
            f"Evaluation '{summary_path}' has conditions {sorted(condition_names)}; "
            f"expected {list(CONDITION_ORDER)}."
        )

    condition_metrics = _read_csv(run_directory / "condition_metrics.csv")
    robustness_summary = _read_csv(run_directory / "robustness_summary.csv")
    if len(condition_metrics) != len(CONDITION_ORDER):
        raise ResultAggregationError(
            f"Expected four condition rows in {run_directory / 'condition_metrics.csv'}."
        )
    if len(robustness_summary) != 1:
        raise ResultAggregationError(
            f"Expected one robustness row in {run_directory / 'robustness_summary.csv'}."
        )

    architecture = str(summary.get("architecture", ""))
    training_condition = str(summary.get("training_condition", ""))
    experiment_id = str(summary.get("experiment_id", ""))
    per_class_tables: list[pd.DataFrame] = []
    predictions: dict[str, pd.DataFrame] = {}
    for condition in CONDITION_ORDER:
        condition_directory = run_directory / "conditions" / condition
        per_class = _read_csv(condition_directory / "per_class_metrics.csv")
        per_class.insert(0, "snr_db", CONDITION_SNR[condition])
        per_class.insert(0, "condition", condition)
        per_class.insert(0, "training_condition", training_condition)
        per_class.insert(0, "architecture", architecture)
        per_class.insert(0, "experiment_id", experiment_id)
        per_class_tables.append(per_class)
        predictions[condition] = _read_csv(
            condition_directory / "predictions.csv"
        )

    return _RunArtifacts(
        summary_path=summary_path,
        summary=summary,
        protocol=protocol,
        condition_metrics=condition_metrics,
        robustness_summary=robustness_summary,
        per_class_metrics=pd.concat(per_class_tables, ignore_index=True),
        predictions=predictions,
    )


def _discover_runs(input_directory: str | Path) -> list[_RunArtifacts]:
    """Discover exactly one final result for every required experiment group."""
    root = Path(input_directory).expanduser().resolve()
    if not root.is_dir():
        raise ResultAggregationError(
            f"Robustness input directory does not exist: {root}"
        )
    summary_paths = sorted(root.glob("*/test/evaluation_summary.json"))
    if not summary_paths:
        raise ResultAggregationError(
            f"No final evaluation summaries found beneath: {root}"
        )
    runs = [_load_run(path) for path in summary_paths]
    observed: dict[tuple[str, str], _RunArtifacts] = {}
    for run in runs:
        group = (
            str(run.summary.get("architecture", "")),
            str(run.summary.get("training_condition", "")),
        )
        if group in observed:
            raise ResultAggregationError(
                f"Duplicate final evaluation for group {group}: "
                f"{observed[group].summary_path} and {run.summary_path}."
            )
        observed[group] = run
    missing = EXPECTED_GROUPS - set(observed)
    unexpected = set(observed) - EXPECTED_GROUPS
    if missing or unexpected:
        raise ResultAggregationError(
            "Final evaluation matrix is incomplete or unexpected; "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}."
        )
    return [observed[group] for group in sorted(EXPECTED_GROUPS)]


def _protocol_signature(run: _RunArtifacts) -> dict[str, Any]:
    """Return protocol fields that must be identical across every model."""
    return {
        "split": run.protocol.get("split"),
        "folds": run.protocol.get("folds"),
        "evaluation": run.protocol.get("evaluation"),
        "resolved_noise_directory": run.protocol.get("resolved_noise_directory"),
        "noise_file_count": run.protocol.get("noise_file_count"),
        "sample_limit": run.protocol.get("sample_limit"),
        "num_samples": run.summary.get("num_samples"),
    }


def _assert_frame_equal(
    current: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    description: str,
) -> None:
    """Raise a research-specific error for mismatched protocol tables."""
    try:
        pd.testing.assert_frame_equal(
            current.reset_index(drop=True),
            reference.reset_index(drop=True),
            check_dtype=False,
        )
    except AssertionError as error:
        raise ResultAggregationError(
            f"Evaluation protocol mismatch in {description}: {error}"
        ) from error


def _validate_protocols(runs: list[_RunArtifacts]) -> dict[str, Any]:
    """Prove all models saw identical labels, noise files, and noise segments."""
    reference_signature = _protocol_signature(runs[0])
    if reference_signature["split"] != "test":
        raise ResultAggregationError("Aggregated protocol must use the test split.")
    if reference_signature["sample_limit"] is not None:
        raise ResultAggregationError("Final evaluations cannot use sample limits.")

    reference_targets: pd.DataFrame | None = None
    reference_noise: pd.DataFrame | None = None
    for run in runs:
        if _protocol_signature(run) != reference_signature:
            raise ResultAggregationError(
                "Evaluation protocol differs for "
                f"{run.summary.get('experiment_id')}."
            )
        expected_samples = int(run.summary["num_samples"])
        for condition in CONDITION_ORDER:
            predictions = run.predictions[condition]
            required = {
                "sample_id",
                "target",
                "condition",
                "target_snr_db",
                "achieved_snr_db",
                "noise_path",
                "noise_selection_seed",
                "noise_applied",
            }
            missing = required - set(predictions.columns)
            if missing:
                raise ResultAggregationError(
                    f"Predictions for {run.summary_path} / {condition} "
                    f"are missing columns: {sorted(missing)}."
                )
            if len(predictions) != expected_samples:
                raise ResultAggregationError(
                    f"Prediction count mismatch for {run.summary_path} / {condition}."
                )
            targets = predictions.loc[:, ["sample_id", "target"]]
            if reference_targets is None:
                reference_targets = targets
            else:
                _assert_frame_equal(
                    targets,
                    reference_targets,
                    description=(
                        f"sample IDs/targets for {run.summary_path} / {condition}"
                    ),
                )

            if condition == "clean":
                if predictions["target_snr_db"].notna().any():
                    raise ResultAggregationError(
                        f"Clean predictions contain target SNR: {run.summary_path}"
                    )
                continue

            target_snr = float(CONDITION_SNR[condition])
            observed_targets = pd.to_numeric(
                predictions["target_snr_db"],
                errors="coerce",
            )
            if not np.allclose(observed_targets, target_snr, atol=0.0, rtol=0.0):
                raise ResultAggregationError(
                    f"Target SNR mismatch for {run.summary_path} / {condition}."
                )
            applied = predictions["noise_applied"].astype(str).str.lower() == "true"
            achieved = pd.to_numeric(
                predictions.loc[applied, "achieved_snr_db"],
                errors="coerce",
            )
            if achieved.isna().any() or (
                (achieved - target_snr).abs() > 1.0e-4
            ).any():
                raise ResultAggregationError(
                    f"Achieved SNR exceeds tolerance for "
                    f"{run.summary_path} / {condition}."
                )
            noise_assignment = predictions.loc[
                :,
                ["sample_id", "noise_path", "noise_selection_seed"],
            ]
            if reference_noise is None:
                reference_noise = noise_assignment
            else:
                _assert_frame_equal(
                    noise_assignment,
                    reference_noise,
                    description=(
                        f"noise assignments for {run.summary_path} / {condition}"
                    ),
                )
    return reference_signature


def _sort_condition_rows(table: pd.DataFrame) -> pd.DataFrame:
    """Sort architecture/training rows using the research condition order."""
    ordered = table.copy()
    ordered["_condition_order"] = ordered["condition"].map(
        {name: index for index, name in enumerate(CONDITION_ORDER)}
    )
    sort_columns = [
        column
        for column in (
            "architecture",
            "training_condition",
            "_condition_order",
            "label",
        )
        if column in ordered.columns
    ]
    ordered = ordered.sort_values(sort_columns).drop(columns="_condition_order")
    return ordered.reset_index(drop=True)


def _build_augmentation_effects(condition_metrics: pd.DataFrame) -> pd.DataFrame:
    """Create baseline/augmented values and signed deltas per condition."""
    keys = ["architecture", "condition", "snr_db"]
    metric_names = (
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "accuracy_drop",
        "macro_f1_drop",
    )
    available_metrics = [
        name
        for name in metric_names
        if name in condition_metrics.columns
    ]
    baseline = condition_metrics.loc[
        condition_metrics["training_condition"] == "baseline",
        [*keys, *available_metrics],
    ].copy()
    augmented = condition_metrics.loc[
        condition_metrics["training_condition"] == "augmented",
        [*keys, *available_metrics],
    ].copy()
    baseline = baseline.rename(
        columns={name: f"baseline_{name}" for name in available_metrics}
    )
    augmented = augmented.rename(
        columns={name: f"augmented_{name}" for name in available_metrics}
    )
    comparison = baseline.merge(
        augmented,
        on=keys,
        how="outer",
        validate="one_to_one",
    )
    if len(comparison) != 3 * len(CONDITION_ORDER):
        raise ResultAggregationError(
            "Baseline/augmented condition pairing is incomplete."
        )
    for metric_name in available_metrics:
        comparison[f"{metric_name}_delta"] = (
            comparison[f"augmented_{metric_name}"]
            - comparison[f"baseline_{metric_name}"]
        )
        comparison[f"{metric_name}_delta_percentage_points"] = (
            comparison[f"{metric_name}_delta"] * 100.0
        )
    return _sort_condition_rows(comparison)


def _build_robustness_effects(
    robustness_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Compare model-level robustness summaries by training condition."""
    metrics = (
        "clean_accuracy",
        "clean_macro_f1",
        "normalized_accuracy_snr_auc",
        "normalized_macro_f1_snr_auc",
        "worst_noisy_accuracy",
        "worst_noisy_macro_f1",
    )
    baseline = robustness_summary.loc[
        robustness_summary["training_condition"] == "baseline",
        ["architecture", *metrics],
    ].rename(columns={name: f"baseline_{name}" for name in metrics})
    augmented = robustness_summary.loc[
        robustness_summary["training_condition"] == "augmented",
        ["architecture", *metrics],
    ].rename(columns={name: f"augmented_{name}" for name in metrics})
    comparison = baseline.merge(
        augmented,
        on="architecture",
        how="outer",
        validate="one_to_one",
    )
    if len(comparison) != 3:
        raise ResultAggregationError("Model-level augmentation pairing is incomplete.")
    for metric_name in metrics:
        comparison[f"{metric_name}_delta"] = (
            comparison[f"augmented_{metric_name}"]
            - comparison[f"baseline_{metric_name}"]
        )
    return comparison.sort_values("architecture").reset_index(drop=True)


def _build_per_class_effects(per_class_metrics: pd.DataFrame) -> pd.DataFrame:
    """Create baseline/augmented class-level metric deltas."""
    keys = ["architecture", "condition", "snr_db", "label", "class_name"]
    metric_names = ("precision", "recall", "f1", "support")
    baseline = per_class_metrics.loc[
        per_class_metrics["training_condition"] == "baseline",
        [*keys, *metric_names],
    ].copy()
    augmented = per_class_metrics.loc[
        per_class_metrics["training_condition"] == "augmented",
        [*keys, *metric_names],
    ].copy()
    baseline = baseline.rename(
        columns={name: f"baseline_{name}" for name in metric_names}
    )
    augmented = augmented.rename(
        columns={name: f"augmented_{name}" for name in metric_names}
    )
    comparison = baseline.merge(
        augmented,
        on=keys,
        how="outer",
        validate="one_to_one",
    )
    for metric_name in ("precision", "recall", "f1"):
        comparison[f"{metric_name}_delta"] = (
            comparison[f"augmented_{metric_name}"]
            - comparison[f"baseline_{metric_name}"]
        )
    return _sort_condition_rows(comparison)


def aggregate_evaluation_results(
    input_directory: str | Path,
) -> AggregatedResults:
    """Load, validate, and combine the complete six-run result matrix."""
    runs = _discover_runs(input_directory)
    protocol = _validate_protocols(runs)
    condition_metrics = _sort_condition_rows(
        pd.concat([run.condition_metrics for run in runs], ignore_index=True)
    )
    robustness_summary = pd.concat(
        [run.robustness_summary for run in runs],
        ignore_index=True,
    ).sort_values(["architecture", "training_condition"]).reset_index(drop=True)
    per_class_metrics = _sort_condition_rows(
        pd.concat([run.per_class_metrics for run in runs], ignore_index=True)
    )

    condition_metrics["accuracy_rank_within_condition"] = condition_metrics.groupby(
        "condition"
    )["accuracy"].rank(method="min", ascending=False)
    condition_metrics["macro_f1_rank_within_condition"] = condition_metrics.groupby(
        "condition"
    )["macro_f1"].rank(method="min", ascending=False)
    robustness_summary["macro_f1_auc_rank"] = robustness_summary[
        "normalized_macro_f1_snr_auc"
    ].rank(method="min", ascending=False)

    return AggregatedResults(
        condition_metrics=condition_metrics,
        robustness_summary=robustness_summary,
        robustness_effects=_build_robustness_effects(robustness_summary),
        augmentation_effects=_build_augmentation_effects(condition_metrics),
        per_class_metrics=per_class_metrics,
        per_class_effects=_build_per_class_effects(per_class_metrics),
        protocol=protocol,
    )


def _model_label(architecture: str, training_condition: str) -> str:
    """Return a compact publication label for one trained model."""
    architecture_label = {
        "cnn": "CNN",
        "crnn": "CRNN",
        "resnet18": "ResNet18",
    }.get(architecture, architecture)
    return f"{architecture_label} {training_condition}"


def _plot_condition_curves(
    condition_metrics: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """Plot one performance metric across ordered noise conditions."""
    colors = {
        "cnn": "#1f77b4",
        "crnn": "#ff7f0e",
        "resnet18": "#2ca02c",
    }
    figure, axis = plt.subplots(figsize=(9.5, 5.8), constrained_layout=True)
    x_values = np.arange(len(CONDITION_ORDER))
    for (architecture, training_condition), group in condition_metrics.groupby(
        ["architecture", "training_condition"],
        sort=True,
    ):
        indexed = group.set_index("condition")
        values = [float(indexed.loc[name, metric]) for name in CONDITION_ORDER]
        axis.plot(
            x_values,
            values,
            marker="o",
            linewidth=2.2,
            linestyle="-" if training_condition == "augmented" else "--",
            color=colors.get(architecture),
            label=_model_label(architecture, training_condition),
        )
    axis.set(
        xticks=x_values,
        xticklabels=CONDITION_LABELS,
        xlabel="Evaluation condition",
        ylabel=ylabel,
        ylim=(0.0, 1.0),
        title=f"{ylabel} under held-out MS-SNSD noise",
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=2, fontsize=9)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_augmentation_effects(
    effects: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot augmented-minus-baseline macro-F1 in percentage points."""
    architectures = ("cnn", "crnn", "resnet18")
    colors = ("#1f77b4", "#ff7f0e", "#2ca02c")
    x_values = np.arange(len(CONDITION_ORDER))
    width = 0.24
    figure, axis = plt.subplots(figsize=(9.5, 5.8), constrained_layout=True)
    for index, (architecture, color) in enumerate(zip(architectures, colors)):
        group = effects.loc[effects["architecture"] == architecture].set_index(
            "condition"
        )
        values = [
            float(group.loc[name, "macro_f1_delta_percentage_points"])
            for name in CONDITION_ORDER
        ]
        axis.bar(
            x_values + (index - 1) * width,
            values,
            width,
            color=color,
            label=architecture.upper() if architecture != "resnet18" else "ResNet18",
        )
    axis.axhline(0.0, color="black", linewidth=0.9)
    axis.set(
        xticks=x_values,
        xticklabels=CONDITION_LABELS,
        xlabel="Evaluation condition",
        ylabel="Macro-F1 change (percentage points)",
        title="Effect of augmentation relative to each architecture's baseline",
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_robustness_auc(
    robustness_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot normalized macro-F1 SNR area by model and training condition."""
    architectures = ("cnn", "crnn", "resnet18")
    x_values = np.arange(len(architectures))
    width = 0.34
    figure, axis = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    for index, (condition, color) in enumerate(
        (("baseline", "#7f7f7f"), ("augmented", "#2ca02c"))
    ):
        values = []
        for architecture in architectures:
            row = robustness_summary.loc[
                (robustness_summary["architecture"] == architecture)
                & (robustness_summary["training_condition"] == condition)
            ].iloc[0]
            values.append(float(row["normalized_macro_f1_snr_auc"]))
        axis.bar(
            x_values + (index - 0.5) * width,
            values,
            width,
            color=color,
            label=condition.capitalize(),
        )
    axis.set(
        xticks=x_values,
        xticklabels=("CNN", "CRNN", "ResNet18"),
        ylabel="Normalized macro-F1 SNR AUC",
        ylim=(0.0, 1.0),
        title="Overall robustness across noisy conditions",
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_zero_db_class_effects(
    per_class_effects: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot class-level augmentation gains at the hardest noise condition."""
    zero_db = per_class_effects.loc[
        per_class_effects["condition"] == "snr_0db"
    ].copy()
    class_order = (
        zero_db.loc[:, ["label", "class_name"]]
        .drop_duplicates()
        .sort_values("label")["class_name"]
        .tolist()
    )
    architectures = ("cnn", "crnn", "resnet18")
    matrix = (
        zero_db.pivot(
            index="architecture",
            columns="class_name",
            values="f1_delta",
        )
        .reindex(index=architectures, columns=class_order)
        .to_numpy(dtype=float)
        * 100.0
    )
    maximum = max(1.0, float(np.nanmax(np.abs(matrix))))
    figure, axis = plt.subplots(figsize=(13.0, 4.3), constrained_layout=True)
    image = axis.imshow(
        matrix,
        aspect="auto",
        cmap="coolwarm",
        vmin=-maximum,
        vmax=maximum,
    )
    axis.set(
        xticks=np.arange(len(class_order)),
        xticklabels=class_order,
        yticks=np.arange(len(architectures)),
        yticklabels=("CNN", "CRNN", "ResNet18"),
        title="0 dB class-level macro-F1 effect of augmentation",
        xlabel="UrbanSound8K class",
    )
    axis.tick_params(axis="x", rotation=35, labelsize=8)
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(
                column_index,
                row_index,
                f"{matrix[row_index, column_index]:+.1f}",
                ha="center",
                va="center",
                fontsize=7,
            )
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("F1 change (percentage points)")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_aggregated_results(
    results: AggregatedResults,
    output_directory: str | Path,
) -> dict[str, Path]:
    """Persist all master tables, figures, protocol checks, and key findings."""
    output = Path(output_directory).expanduser().resolve()
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    paths = {
        "condition_metrics": output / "master_condition_metrics.csv",
        "robustness_summary": output / "model_robustness_summary.csv",
        "robustness_effects": output / "robustness_augmentation_effects.csv",
        "augmentation_effects": output / "condition_augmentation_effects.csv",
        "per_class_metrics": output / "master_per_class_metrics.csv",
        "per_class_effects": output / "per_class_augmentation_effects.csv",
        "protocol": output / "aggregation_protocol.json",
        "summary": output / "analysis_summary.json",
        "accuracy_curve": figures / "accuracy_robustness_curves.png",
        "macro_f1_curve": figures / "macro_f1_robustness_curves.png",
        "augmentation_plot": figures / "macro_f1_augmentation_effects.png",
        "robustness_auc_plot": figures / "macro_f1_robustness_auc.png",
        "zero_db_class_plot": figures / "zero_db_per_class_augmentation_effects.png",
    }
    results.condition_metrics.to_csv(paths["condition_metrics"], index=False)
    results.robustness_summary.to_csv(paths["robustness_summary"], index=False)
    results.robustness_effects.to_csv(paths["robustness_effects"], index=False)
    results.augmentation_effects.to_csv(paths["augmentation_effects"], index=False)
    results.per_class_metrics.to_csv(paths["per_class_metrics"], index=False)
    results.per_class_effects.to_csv(paths["per_class_effects"], index=False)
    paths["protocol"].write_text(
        json.dumps(results.protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    _plot_condition_curves(
        results.condition_metrics,
        metric="accuracy",
        ylabel="Accuracy",
        output_path=paths["accuracy_curve"],
    )
    _plot_condition_curves(
        results.condition_metrics,
        metric="macro_f1",
        ylabel="Macro-F1",
        output_path=paths["macro_f1_curve"],
    )
    _plot_augmentation_effects(
        results.augmentation_effects,
        paths["augmentation_plot"],
    )
    _plot_robustness_auc(
        results.robustness_summary,
        paths["robustness_auc_plot"],
    )
    _plot_zero_db_class_effects(
        results.per_class_effects,
        paths["zero_db_class_plot"],
    )

    best_by_condition: dict[str, dict[str, object]] = {}
    for condition in CONDITION_ORDER:
        candidates = results.condition_metrics.loc[
            results.condition_metrics["condition"] == condition
        ]
        best = candidates.sort_values("macro_f1", ascending=False).iloc[0]
        best_by_condition[condition] = {
            "architecture": str(best["architecture"]),
            "training_condition": str(best["training_condition"]),
            "accuracy": float(best["accuracy"]),
            "macro_f1": float(best["macro_f1"]),
        }
    best_robustness = results.robustness_summary.sort_values(
        "normalized_macro_f1_snr_auc",
        ascending=False,
    ).iloc[0]
    largest_auc_gain = results.robustness_effects.sort_values(
        "normalized_macro_f1_snr_auc_delta",
        ascending=False,
    ).iloc[0]
    summary = {
        "num_models": int(len(results.robustness_summary)),
        "num_condition_results": int(len(results.condition_metrics)),
        "num_per_class_results": int(len(results.per_class_metrics)),
        "best_macro_f1_by_condition": best_by_condition,
        "best_overall_robustness": {
            "architecture": str(best_robustness["architecture"]),
            "training_condition": str(best_robustness["training_condition"]),
            "normalized_macro_f1_snr_auc": float(
                best_robustness["normalized_macro_f1_snr_auc"]
            ),
        },
        "largest_augmentation_auc_gain": {
            "architecture": str(largest_auc_gain["architecture"]),
            "macro_f1_snr_auc_delta": float(
                largest_auc_gain["normalized_macro_f1_snr_auc_delta"]
            ),
        },
        "files": {
            name: str(path)
            for name, path in paths.items()
            if name not in {"summary"}
        },
    }
    paths["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return paths
