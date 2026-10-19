# 0003: Filters are plain functions that return kept and rejected tables

Status: Accepted (2026-10-18)

## Context
Before resizing, the data needs cleaning: unreadable files, images too small to resize without upscaling, near-blank images, extreme shapes (see ADR 0002), and exact duplicates. The cleaning has to be repeatable, explainable, and safe for the raw data.

## Decision
- `apply_filters(scan_df, FiltersConfig)` in `src/authentiscan/data/filters.py` takes the scan table and returns two tables: kept rows and rejected rows. Each rejected row carries `reject_reason`. When several rules apply, the reasons are joined with `;`.
- Rules and thresholds: decode failure, `min_short_side` 256, `max_aspect_ratio` 3.0, `min_pixel_std` 2.0, and exact duplicates by md5. All thresholds live in `configs/data.yaml` and are validated by Pydantic with `extra="forbid"`.
- Between exact duplicates, a copy in `train` beats a copy in `test`. The test split stays clean, because a test image that is also a train image would inflate the metrics.
- Raw data is never modified, moved, or deleted. Rejected images are listed in a CSV (`excluded_images.csv`) with a reason, not copied to a folder.
- No pipeline class. The steps are plain functions, Pydantic models for data and config, and one PyTorch `Dataset` later.

## Alternatives considered
- **Delete or move bad files.** Rejected, because it cannot be undone and the raw data should stay as downloaded.
- **A pipeline class holding state.** Rejected, because plain functions are easier to test and the steps are short.
- **Filter inside the Dataset at training time.** Rejected, because the rejected set would not be written down anywhere and the counts could change between runs.

## Consequences
- Running the script twice gives byte-identical CSVs (checked with md5), so the pipeline is idempotent.
- On the real data: 59,760 kept and 240 rejected.
- Every threshold change is a config edit and a re-run, with a CSV that shows exactly what changed.
- Open: `min_pixel_std` of 2.0 versus 1.0 affects 4 images and has not been viewed yet. Near-duplicates (copies that are not byte-identical) are not handled here. They need a perceptual hash and are left for the split step.
