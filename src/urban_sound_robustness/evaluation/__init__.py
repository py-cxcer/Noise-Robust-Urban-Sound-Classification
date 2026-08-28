"""Classification metrics and controlled robustness evaluation."""

from urban_sound_robustness.evaluation.aggregation import (
    AggregatedResults,
    ResultAggregationError,
    aggregate_evaluation_results,
    save_aggregated_results,
)
from urban_sound_robustness.evaluation.checkpoint import (
    CheckpointEvaluationError,
    ConditionPredictionCollection,
    ResearchCheckpoint,
    RobustnessEvaluationDataset,
    collect_condition_predictions,
    load_research_checkpoint,
    validate_noise_isolation,
)
from urban_sound_robustness.evaluation.corruption import (
    ControlledCorruption,
    DeterministicNoiseCorruptor,
    NoiseDatasetError,
    RobustnessCondition,
    discover_noise_files,
    parse_robustness_conditions,
    stable_seed,
)
from urban_sound_robustness.evaluation.metrics import (
    ClassificationResult,
    calculate_classification_metrics,
    collect_model_predictions,
    save_classification_result,
)
from urban_sound_robustness.evaluation.robustness import (
    RobustnessAnalysis,
    calculate_robustness_metrics,
    save_robustness_analysis,
)

__all__ = [
    "AggregatedResults",
    "CheckpointEvaluationError",
    "ControlledCorruption",
    "ClassificationResult",
    "ConditionPredictionCollection",
    "DeterministicNoiseCorruptor",
    "NoiseDatasetError",
    "RobustnessCondition",
    "RobustnessAnalysis",
    "RobustnessEvaluationDataset",
    "ResearchCheckpoint",
    "ResultAggregationError",
    "aggregate_evaluation_results",
    "calculate_classification_metrics",
    "calculate_robustness_metrics",
    "collect_model_predictions",
    "collect_condition_predictions",
    "discover_noise_files",
    "parse_robustness_conditions",
    "load_research_checkpoint",
    "save_classification_result",
    "save_aggregated_results",
    "save_robustness_analysis",
    "stable_seed",
    "validate_noise_isolation",
]
