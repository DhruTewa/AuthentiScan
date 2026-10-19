# 0004: Center-crop to a square once, normalize and augment at training time

Status: Accepted (2026-10-19)

## Context
Images have different shapes, and the model needs a fixed square input. Wide images (up to aspect ratio 3.0 after filtering) have to be turned into squares somehow. Shape also differs by class (ADR 0002), so the way we do it must not add a class-dependent artefact.

## Decision
- **Stored once, on disk:** EXIF rotation, RGB conversion, short side resized to 256 (bicubic), center-crop to 256x256, saved as PNG.
- **At training time, in the Dataset:** crop to 224 (random for train, center for eval and serving), scale to 0..1, normalize with the mean and std of the train split only.
- One function, `build_transform`, builds this pipeline, so serving applies exactly what evaluation applies.
- Baseline augmentation is the random crop only. A horizontal flip is a later experiment. Blur, noise, color changes, JPEG compression and small rotations are avoided.

## Alternatives considered
- **Squash the whole image to 256x256.** Rejected: it stretches wide images, and which images get stretched depends on class.
- **Pad to a square.** Rejected: the padding becomes a pattern the model can learn, and how much padding an image gets differs by class.
- **Keep original sizes and crop while training.** Rejected: every epoch would re-read and decode about 52 GB.
- **Save a random crop to disk.** Rejected: baked-in randomness is hard to reproduce, and augmentation belongs in the Dataset.
- **Normalize on disk.** Not possible with PNG, which stores whole numbers from 0 to 255, and the stats must come from the train split only.
- **Compute the stats inside preprocessing.** Rejected: preprocessing skips existing files, so a resumed run would miss images, and the stats would go stale if the split changed.

## Consequences
- Center-crop throws away the edges of wide images. The model never sees the full panorama.
- Stats are computed once from the stored 256 images, in a separate script (`scripts/compute_norm_stats.py`), and saved to `data/processed/norm_stats.json`. Both training and serving load that file.
- Rotation and other resampling augmentations are excluded because they smooth pixels and could erase the faint traces of generation the model needs.
- With only a random crop, overfitting may appear sooner. If it does, the flip is the first thing to try.
- `crop_size` must not exceed `image_size`. The config validator enforces it, because torchvision would otherwise fail or pad.
