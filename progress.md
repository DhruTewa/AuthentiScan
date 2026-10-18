# Progress Log

Reverse-chronological, append-only. New entries go at the top. Never edit past entries.

## 2026-10-16

**Decided**
- Output format is lossless PNG (JPEG 95 was briefly chosen, then reversed). `jpeg_quality` removed from the config.
- Config section is named `filters`, not `validate`, because Pydantic's `BaseModel.validate` would be shadowed.
- Package name is `authentiscan` (the 2026-10-14 entry says `aivsreal`).
- No pipeline class: plain functions, Pydantic models, and one PyTorch `Dataset` subclass. Rejected images go to a CSV with a reason, not a copied folder. Raw data is never modified.
- Filter rules: short side at least 256, aspect ratio at most 3.0, pixel std at least 2.0, exact duplicates removed (train copy beats test copy).

**Implemented**
- `configs/data.yaml`, `src/authentiscan/schemas.py` (Pydantic, `extra="forbid"`), `src/authentiscan/data/filters.py`, and 30 passing tests.
- On the real scan, the filters keep 59,760 images and reject 240 (163 fake, 77 real).
- Repo pushed to GitHub (`DhruTewa/AuthentiScan`). README written with project description and EDA findings.

**Open questions**
- The 4 low-contrast images still need viewing (`min_pixel_std` 2 vs 1).
- Rejects skew toward fakes, mostly from the aspect-ratio rule. Needs an ADR.
- Near-duplicates are unhandled until the split step.
- Keep or drop the extra dependencies in `pyproject.toml` (`fastai`, `directory-tree`, `imageio`, `requests`, `seaborn`, `ipywidgets`).
- Dataset error policy: log skips, count them, stop if too many.
- `notebooks/scan.py` is a duplicate and can be deleted once the package version is finished.

**Next session goal**
- Harden `src/authentiscan/data/scan.py` (already in the package as a copy of the notebook script): typed `ScanRow`, config paths, logging, a fix for `stat()` outside the try block, and tests. Then write the runner that saves `manifest.csv` and the rejects file.

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
