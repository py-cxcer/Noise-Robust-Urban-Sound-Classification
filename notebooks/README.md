# Notebooks

Notebooks are reserved for dataset exploration, audio visualization, error
analysis, and report-quality figures. Reusable processing and training logic must
remain in `src/urban_sound_robustness/` and be imported from notebooks.

## UrbanSound8K EDA

`01_urbansound8k_eda.ipynb` is an executed, reproducible analysis of the complete
dataset. It covers integrity, class and fold distributions, duration, source
sample rates and channels, representative waveforms, Log-Mel spectrograms, and
MFCCs. It saves research figures to `results/figures/eda/` and supporting CSV/JSON
files to `results/metrics/eda/`.

After activating `.venv`, open the notebook interactively with:

```powershell
python -m jupyter lab notebooks\01_urbansound8k_eda.ipynb
```

Re-execute every cell non-interactively and update the stored outputs with:

```powershell
python -m jupyter nbconvert --to notebook --execute --inplace notebooks\01_urbansound8k_eda.ipynb --ExecutePreprocessor.timeout=600
```
