# Local data layout

- `raw/`: downloaded datasets in their original structure.
- `processed/`: optional cached features or derived metadata.
- `external_noise/`: noise-only audio used for augmentation and test corruption.
- `metadata/`: project-generated metadata that does not belong inside a dataset.

Dataset and audio contents are intentionally ignored by Git. External noise files
must never be exposed as classification targets.

The deterministic corruption loader searches `external_noise/` recursively for
`.wav`, `.flac`, and `.ogg` files. This directory is intentionally empty in the
repository. Add a properly licensed noise-only collection before research
evaluation; seeded synthetic white noise is used only by the inspection CLI to
verify mathematics and is not a substitute for the research noise dataset.

UrbanSound8K is expected at:

```text
raw/UrbanSound8K/audio/fold1 ... fold10
raw/UrbanSound8K/metadata/UrbanSound8K.csv
```

The official version 1.0 archive is published at DOI `10.5281/zenodo.1203745`.
When the Zenodo transfer is impractically slow, the reproducible setup uses the
`danavery/urbansound8K` Hugging Face mirror. Its lossless Parquet shards contain
the WAV bytes, official filenames, folds, and metadata. The preparation script
reconstructs the standard layout without decoding or resampling and validates all
8,732 records. Kaggle's `chrisfilo/urbansound8k` mirror remains an archive fallback.
Preserve and follow the original dataset license and attribution.
