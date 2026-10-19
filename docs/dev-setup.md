# Developer setup

## Requirements
- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/) (tested with 0.11.9)
- Apple Silicon Mac for training on MPS (preprocessing and tests run on any CPU)
- About 60 GB of free disk: roughly 52 GB of raw data plus 6 GB of processed images

## Install
```
uv sync
```
This creates `.venv/` from `pyproject.toml` and `uv.lock`. Run everything through `uv run` so the right environment is used.

## Data
Put the dataset in `data/` (gitignored) so that images sit at `data/<split>/<label>/<file>`, where split is `train` or `test` and label is `fake` or `real`. Labels come from the folder names. Raw data is never modified.

## Pipeline
Run from the project root. All paths and thresholds come from `configs/data.yaml`.

| Step | Command | Writes |
|---|---|---|
| 1. Scan, filter, split | `uv run python scripts/prepare_dataset.py` | `data/interim/` (scan, dataset manifest, excluded images, split manifest) |
| 2. Resize to 256x256 PNG | `uv run python scripts/preprocess_images.py` | `data/processed/` (images and `manifest.csv`) |
| 3. Normalization stats (train split only) | `uv run python scripts/compute_norm_stats.py` | `data/processed/norm_stats.json` |
| 4. Look at a real training batch | `uv run python scripts/inspect_batch.py` | `data/interim/batch_check.png` |

All steps are safe to re-run (steps 3 and 4 simply overwrite their output). The scan is cached in `data/interim/scan.csv`, so delete that file to force a fresh scan. Preprocessing skips images that already exist, so an interrupted run resumes where it stopped. Step 3 depends on steps 1 and 2, and step 4 on all three.

## Design decisions
The reasons behind the main choices (PNG output, aspect-ratio filter, filter design, center-crop and training-time transforms) are in `docs/decisions/`.

## Tests
```
uv run pytest -q
```
Tests build their own small images in a temporary folder, so they need no real data.

## Dependencies
Adding a dependency means updating `pyproject.toml` (with `uv add <package>`) and adding a row here.

| Package | Used for |
|---|---|
| pillow | Reading, resizing, and saving images |
| numpy, pandas | Scan statistics and manifests |
| pydantic, pyyaml | Config and row schemas |
| tqdm | Progress bar during preprocessing |
| pytest | Tests |
| torch, torchvision, torchmetrics | Training and evaluation (next phase) |
| matplotlib | EDA notebook plots |
| optuna, fastapi, mlflow | Tuning, serving, and tracking (later phases) |

Declared but not used by `src/`, `scripts/`, or `tests/`: `fastai`, `directory-tree`, `imageio`, `requests`, `seaborn`, `ipywidgets`. Only `notebooks/helper_utils.py` mentions them, and it imports them only if installed. Whether to keep them is an open decision (see `progress.md`).
