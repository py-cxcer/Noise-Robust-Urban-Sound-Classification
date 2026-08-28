# Tests

The current suite tests configuration loading and validation, path resolution,
device selection, deterministic seeds, logging, experiment IDs, directory isolation,
reproducibility snapshots, UrbanSound8K metadata integrity, fold filtering, missing
files, corrupt audio, header inspection, result serialization, and validated
channels-first waveform loading. Deterministic mono conversion tests cover channel
averaging, input immutability, dtype preservation, gradient flow, and invalid
waveform rejection. The suite also covers up/downsampling, exact padding and
cropping, seeded random crops, log-Mel shapes, silence stability, composed
preprocessing, reusable MFCC extraction, lazy dataset loading, DataLoader
collation, exact target-SNR mixing, silence policies, short/long noise handling,
recursive noise discovery, deterministic per-sample corruption, independently
switchable augmentation, validation/test augmentation bypass, CNN/CRNN/ResNet18
output shapes, loss/backpropagation, trainer history, checkpoints, classification
metrics, result storage, and robustness summaries. The current suite contains
121 passing tests. Checkpoint-evaluation coverage additionally rejects smoke and
non-best checkpoints, permits only the live evaluation protocol to override old
snapshots, verifies disjoint training/test noise banks, and preserves achieved
SNR provenance through inference. Run it verbosely
with:

```bash
python -m pytest -v
```

Real-audio training and robustness smoke runs are intentionally CLI verifications
rather than unit tests because UrbanSound8K and MS-SNSD files are not committed.
An additional deterministic test proves that interrupted/resumed training
produces exactly the same parameters and history as uninterrupted training.
Aggregation tests require all six architecture/training variants, verify the
generated comparison tables and figures, and reject mismatched test-noise
assignments before research results are combined.
