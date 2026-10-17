# AuthentiScan (AI vs Real Image Classification)

Global rules in `~/.claude/CLAUDE.md` apply. This file only adds project context.

## Goal
Binary classifier (AI-generated vs real images), built end to end the way a real team would: data pipeline, training, evaluation, FastAPI serving in Docker, monitoring.

## Working mode
- Owner writes all code. Claude guides step by step and writes code only when asked.
- Each step: what to create, why, what it connects to, a "done when" check.

## Stack
- Python 3.12, uv, PyTorch (MPS on M1 Pro Max, 64GB RAM)
- torchvision, torchmetrics, optuna (tuning comes after the baseline works)
- Also installed: pydantic, pyyaml, pillow, pytest, fastapi, mlflow (use each only when its phase needs it)

## Data
- Lives in `data/` (gitignored), ~52GB raw, 60,000 images total. Labels come from folder names (`fake`, `real`).
  - `data/train`: 24,000 fake + 24,000 real. `data/test`: 6,000 fake + 6,000 real. Balanced.
- Images are high resolution (~0.8MB each). Plan: resize once into `data/processed/`, train on that.
- Known leaks (see the EDA summary in `notebooks/01_eda.ipynb`): ~36% of fakes are PNG vs ~0.1% of reals, resolution and aspect ratio differ by class, JPEG quality tables differ, ~30% of fakes are palette-mode images. The resize step must re-encode every image identically, and metrics are reported sliced by format and mode. Use actual `format`, not file extension.
- Never commit data or model weights.

## Target layout
Code in `src/authentiscan/` (data, models, training, inference, serving, monitoring, utils), thin CLI entry points in `scripts/`, YAML in `configs/`, tests in `tests/`, ADRs in `docs/decisions/`.

## Project rules
- Preprocessing used in serving must match evaluation preprocessing exactly.
- Splits are seeded and deterministic. Watch for leakage and near-duplicates.
- Evaluate on a generator not seen in training when possible.
- Adding a dependency: update `pyproject.toml` and `docs/dev-setup.md`, and ask first.
