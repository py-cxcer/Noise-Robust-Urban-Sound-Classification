# Experiment configurations

Each run manifest selects dataset, audio, augmentation, model, training, evaluation,
and path component files. Its optional `overrides` mapping changes only settings
specific to that run. `development.yaml` currently provides the low-cost pipeline
development configuration.

Later model phases will add the six primary experiment definitions:

- CNN baseline and augmented
- CRNN baseline and augmented
- ResNet18 baseline and augmented

Keeping composed run files here makes each experiment explicit without duplicating
large blocks of shared settings.
