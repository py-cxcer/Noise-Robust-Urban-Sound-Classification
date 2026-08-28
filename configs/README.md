# Configuration layout

Configuration is separated by responsibility so experiments can change one concern
without duplicating unrelated settings. The loader combines these YAML files into
one validated experiment configuration.

- `dataset/`: dataset adapter settings and official fold assignments.
- `audio/`: shared waveform and feature-extraction settings.
- `augmentation/`: baseline, augmented, and future ablation conditions.
- `model/`: architecture-specific parameters.
- `training/`: optimizer and training-loop behavior.
- `evaluation/`: deterministic corruption conditions and reported metrics.
- `paths/`: operating-system-independent, repository-relative paths.
- `experiment/`: manifests that select one configuration from each group and may
  supply focused nested overrides.

`experiment/development.yaml` is the initial inexpensive smoke configuration.
Run `python scripts/check_config.py` from the repository root to validate and
summarize it. Loading configuration does not load a dataset or create a model.
