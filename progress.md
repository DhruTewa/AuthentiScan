# Progress Log

Reverse-chronological, append-only. New entries go at the top. Never edit past entries.

## 2026-10-14

**Decided**
- Scope: train + serve + monitor. Deployment: FastAPI + Docker, local.
- Framework: PyTorch on Mac MPS. Full dataset, but resized once before training.
- Package layout: `src/aivsreal/`, converted to a package just before the first real module (schemas), not as a separate step.
- DVC / MLflow / Evidently: undecided, each is a new dependency needing approval.

**Implemented**
- uv environment with torch, torchvision, torchmetrics, optuna, pandas, numpy, matplotlib.
- Data downloaded to `data/train` and `data/test` (gitignored).
- `CLAUDE.md` and `progress.md` created.

**Open questions**
- Dataset layout and labels (subfolders vs CSV) not yet inspected.
- Image sizes and class balance unknown.
- Resize target (224 vs 256) and method.
- Data versioning: DVC or a plain manifest file.

**Next session goal**
- Explore the data (`notebooks/01_eda.ipynb`): folder structure, labels, size distribution, class balance.
