# Noise-Robust Urban Sound Classification

A reproducible PyTorch research pipeline for measuring how controlled
background noise affects environmental sound classification and whether
training-time audio augmentation improves robustness.

The completed experiment compares baseline and augmented versions of three
architectures--CNN, CRNN with BiGRU, and ResNet18--on UrbanSound8K under clean,
20 dB, 10 dB, and 0 dB test conditions.

## Project status

The implementation and primary experiment are complete:

- UrbanSound8K integrity, preprocessing, and exploratory analysis
- mathematically controlled signal-to-noise-ratio mixing
- configurable training-only waveform and spectrogram augmentation
- CNN, CRNN, and ResNet18 model families
- six full baseline/augmented training runs
- interruption-safe checkpoints and exact run resumption
- held-out fold-10 evaluation using separated MS-SNSD test noise
- 24 model-condition results and 240 per-class results
- validated cross-model aggregation and research figures
- 121 passing automated tests

The repository contains source code, configurations, tests, the executed EDA
notebook, selected paper figures, and a compact snapshot of the final aggregate
evidence. Raw datasets, checkpoints, logs, and generated runtime artifacts are
excluded from version control.

## Research questions

1. Does training-time audio augmentation reduce classification-performance
   degradation under background noise?
2. Does the augmentation effect differ across CNN, CRNN, and ResNet18?

## Main findings

### Fold-10 macro F1

| Architecture | Training | Clean | 20 dB | 10 dB | 0 dB |
|---|---|---:|---:|---:|---:|
| CNN | Baseline | 0.7323 | 0.6624 | 0.5857 | 0.4230 |
| CNN | Augmented | 0.7469 | 0.7270 | 0.7071 | 0.5481 |
| CRNN | Baseline | **0.8037** | 0.7632 | 0.7134 | 0.4952 |
| CRNN | Augmented | 0.7574 | 0.7634 | **0.7535** | 0.6186 |
| ResNet18 | Baseline | 0.7617 | 0.7050 | 0.5774 | 0.4249 |
| ResNet18 | Augmented | 0.7781 | **0.7783** | 0.7432 | **0.6384** |

### Augmentation effect

Values are augmented minus baseline for the same architecture.

| Architecture | Clean F1 change | 0 dB F1 change | Normalized F1 SNR-AUC change |
|---|---:|---:|---:|
| CNN | +0.0146 | +0.1252 | +0.1082 |
| CRNN | -0.0464 | +0.1234 | +0.0509 |
| ResNet18 | +0.0163 | +0.2135 | +0.1546 |

Augmentation improved noisy-condition robustness for every architecture.
ResNet18 augmented achieved the strongest overall normalized macro-F1 SNR AUC
(0.7258), while CRNN baseline achieved the best clean macro F1 (0.8037).
The CRNN result demonstrates that clean performance and robustness are related
but distinct objectives.

Condition-level, per-class, and aggregate evidence is generated locally under
`results/` by the evaluation and aggregation commands documented below.

## Experimental design

### Dataset split

UrbanSound8K's official folds are preserved:

| Subset | Folds | Samples |
|---|---|---:|
| Training | 1--8 | 7,079 |
| Validation | 9 | 816 |
| Test | 10 | 837 |

The test fold is not used for training, checkpoint selection, or development
smoke tests.

### Audio representation

Every architecture receives the same standardized input:

- mono audio resampled to 22,050 Hz
- four-second clips using zero padding or cropping
- deterministic center crops for validation and test
- Log-Mel spectrograms with 64 Mel bands
- FFT/window length 1,024 and hop length 512
- per-example standardization
- final tensor shape: `[batch, 1, 64, 173]`

### Training conditions

The baseline condition disables robustness-oriented augmentation. The augmented
condition independently applies:

- time shift, probability 0.5, up to 20% of clip length
- random gain, probability 0.5, from -6 dB to +6 dB
- background noise, probability 0.5, at a sampled 0--20 dB SNR
- frequency masking, probability 0.5, up to 8 Mel bins
- time masking, probability 0.5, up to 16 frames

Augmentation is bypassed for validation and test data.

### Robustness evaluation

Each selected `best.pt` checkpoint is evaluated once under:

- clean
- 20 dB SNR
- 10 dB SNR
- 0 dB SNR

Training uses `MS-SNSD/noise_train`; final corruption uses the disjoint
`MS-SNSD/noise_test` split with corruption seed 2025. Sample IDs determine
noise-file and segment selection so every model receives identical test
corruptions.

## Repository layout

```text
configs/                       Composable YAML experiment configuration
data/                          Local dataset/noise placeholders and layout guide
notebooks/                     Executed UrbanSound8K exploratory analysis
scripts/                       Dataset, inspection, training, evaluation CLIs
src/urban_sound_robustness/    Reusable Python package
  audio/                       Loading, resampling, features, SNR mixing
  augmentation/                Waveform and spectrogram augmentation
  datasets/                    Dataset contracts and UrbanSound8K adapter
  evaluation/                  Metrics, checkpoint checks, aggregation
  models/                      CNN, CRNN, ResNet18, model factory
  training/                    Epoch engine, orchestration, recovery
  utils/                       Config, paths, logging, devices, reproducibility
tests/                         Automated project tests
```

Generated content under `data/`, `checkpoints/`, `experiments/`, `logs/`,
and `results/` is intentionally ignored.

## Installation

Python 3.10 or newer is supported. The completed experiments used Python 3.13.1,
PyTorch 2.11.0+cu126, and an NVIDIA GTX 1660 Ti.

Create and activate the required repository-local virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install one PyTorch profile and then the project:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements-cuda.txt
python -m pip install -e .
python -m pip check
```

For a machine without a compatible NVIDIA GPU, use
`requirements-cpu.txt` instead of `requirements-cuda.txt`. Do not install both
profiles.

## Dataset setup

Datasets are not redistributed by this repository. Preserve the licenses and
attribution requirements of UrbanSound8K and the selected external-noise
collection.

The reproducible UrbanSound8K flow downloads a public Parquet snapshot containing
the original WAV bytes and reconstructs the standard directory layout:

```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='danavery/urbansound8K', repo_type='dataset', local_dir='data/raw/UrbanSound8K_parquet')"
python scripts\prepare_urbansound8k_from_parquet.py
python scripts\inspect_dataset.py
```

Expected paths:

```text
data/raw/UrbanSound8K/audio/fold1 ... fold10
data/raw/UrbanSound8K/metadata/UrbanSound8K.csv
data/external_noise/MS-SNSD/noise_train/
data/external_noise/MS-SNSD/noise_test/
```

The noise directories must contain properly licensed noise-only audio. External
noise is used only as augmentation or corruption input and never as a target
class.

## Verification

With `.venv` active:

```powershell
python -m pytest -v
python -m compileall -q src scripts tests
```

Expected test result:

```text
121 passed
```

The suite covers configuration, dataset validation, audio loading and
preprocessing, augmentation isolation, deterministic SNR corruption, model
interfaces, training and exact resume behavior, classification metrics,
checkpoint validation, provenance preservation, and result aggregation.

## Running the pipeline

Validate a manifest:

```powershell
python scripts\check_config.py configs\experiment\cnn_baseline.yaml
```

Run a bounded integration check:

```powershell
python scripts\train.py configs\experiment\development.yaml --epochs 1 --max-train-samples 8 --max-validation-samples 4 --num-workers 0 --run-label smoke
```

Run a full configured experiment:

```powershell
python scripts\train.py configs\experiment\cnn_baseline.yaml --run-label run01
```

Resume an interrupted experiment:

```powershell
python scripts\train.py configs\experiment\cnn_baseline.yaml --resume checkpoints\<experiment-id>\last.pt
```

Evaluate the selected full-run checkpoint on held-out fold 10:

```powershell
python scripts\evaluate_robustness.py configs\experiment\cnn_baseline.yaml checkpoints\<experiment-id>\best.pt
```

Aggregate the six final evaluations:

```powershell
python scripts\aggregate_results.py
```

Detailed CLI behavior and output layouts are documented in
[scripts/README.md](scripts/README.md). Experiment configuration is documented
in [configs/README.md](configs/README.md).

## Reproducibility safeguards

- repository-relative paths through `pathlib.Path`
- configuration and environment snapshots per run
- deterministic Python, NumPy, PyTorch, CUDA, and DataLoader seeds
- official fold preservation
- training-only augmentation
- disjoint training and evaluation noise banks
- independent fixed corruption seed
- stable sample-based corruption assignment
- atomic `best.pt` and `last.pt` checkpoints
- exact optimizer, scheduler, early-stopping, history, and RNG restoration
- strict rejection of smoke or incompatible final checkpoints
- non-overwriting experiment and result directories

## Final report

The completed report is available from both locations:

- [Final submission PDF](CSE437_Group5_Final_Submission.pdf)
- [Project report folder on Google Drive](https://drive.google.com/drive/u/0/folders/1Rn1jU3xm5COCyapM0ZAD2xBiw-Mx_ndc)

## Limitations

- Final testing uses one official UrbanSound8K fold, not full ten-fold
  cross-validation.
- Each training configuration has one completed training seed.
- Confidence intervals and formal paired significance tests are not included.
- Evaluation uses one held-out noise collection and three finite noisy SNR
  points.
- ResNet18 is trained from scratch; pretrained audio transformers are outside
  the experiment scope.

The findings therefore describe this controlled experimental setup and should
not be generalized to every acoustic environment.

## Additional documentation

- [data/README.md](data/README.md): local dataset and noise layout
- [notebooks/README.md](notebooks/README.md): EDA notebook usage
- [scripts/README.md](scripts/README.md): command-line reference
- [tests/README.md](tests/README.md): automated-test scope
