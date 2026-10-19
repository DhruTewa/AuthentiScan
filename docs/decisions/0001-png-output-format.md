# 0001: Write resized images as lossless PNG

Status: Accepted (2026-10-18)

## Context
The raw data leaks the label through file format. In the scan, 35.6% of fake images are PNG against 0.1% of real images, and the real set is 99.9% JPEG. JPEG quality tables also differ by class. A model could learn "PNG means fake" and score well without looking at image content.

The resize step must write every image in one identical format, so the format of the stored file says nothing about the class.

## Decision
Every image is resized once and saved as RGB PNG in `data/processed/`. The config only allows `output_format: PNG`.

## Alternatives considered
- **JPEG at quality 95.** Smaller on disk. It was chosen briefly and then reversed. Re-encoding every image as JPEG adds a second round of compression. Images that were already JPEG would be compressed twice and the former PNGs only once, which is a new difference between the classes.
- **Keep each image's original format.** Rejected, because it keeps the leak.

## Consequences
- Format no longer separates the classes in the processed data.
- PNG files are larger than JPEG, so `data/processed/` needs more disk space and loading is somewhat slower. Disk has room on this machine.
- Metrics are still reported sliced by original format and mode, using the `format` and `mode` columns of the scan, because the content of former PNG and palette images can still differ even after re-encoding.
- Serving must apply the same resize and RGB conversion, so an uploaded JPEG and a PNG are treated the same way.
